from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os
import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

class CharacterSplittingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    def get_strategy_name(self) -> str:
        return "Character Splitting"

class RecursiveCharacterSplittingStrategy(ChunkingStrategy):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def chunk_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
    
    def get_strategy_name(self) -> str:
        return "Recursive Character Text Splitting"

class SemanticChunkingStrategy(ChunkingStrategy):
    def __init__(self, openai_api_key: str):
        self.embedding_model = OpenAIEmbeddings(
            api_key=openai_api_key,
            model="text-embedding-3-small"
        )
        self.chunker = SemanticChunker(
            embeddings=self.embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=80
        )
    
    def chunk_text(self, text: str) -> List[str]:
        return self.chunker.split_text(text)
    
    def get_strategy_name(self) -> str:
        return "Semantic Chunking"

class LLMChunkingStrategy(ChunkingStrategy):
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        
        self.title_summary_prompt = PromptTemplate.from_template("""
        Bạn là một trợ lý AI. Hãy đọc đoạn văn sau và tạo ra:
        - Tiêu đề mô tả ngắn gọn nội dung chính
        - Một đoạn tóm tắt ngắn
        
        Đoạn văn:
        {chunk}
        
        Trả lời theo định dạng sau:
        Title: <tiêu đề>
        Summary: <tóm tắt>
        """)
        
        self.alignment_prompt = PromptTemplate.from_template("""
        Bạn là một trợ lý AI.
        Câu sau có liên quan và nên gộp vào đoạn văn dưới không?
        
        Câu mới:
        "{sentence}"
        
        Đoạn văn hiện tại:
        Tiêu đề: {title}
        Tóm tắt: {summary}
        
        Trả lời "YES" nếu nên gộp, hoặc "NO" nếu không.
        """)
    
    def get_title_summary(self, chunk_text):
        prompt = self.title_summary_prompt.format(chunk=chunk_text)
        resp = self.llm.invoke(prompt).content
        try:
            lines = resp.strip().split('\n')
            title = lines[0].replace("Title:", "").strip()
            summary = lines[1].replace("Summary:", "").strip()
        except:
            title, summary = "No title", "No summary"
        return title, summary
    
    def should_merge(self, sentence, title, summary):
        prompt = self.alignment_prompt.format(sentence=sentence, title=title, summary=summary)
        resp = self.llm.invoke(prompt).content.strip().upper()
        return "YES" in resp
    
    def chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current_chunk = []
        current_title = ""
        current_summary = ""
        
        for sentence in sentences:
            if not current_chunk:
                current_chunk = [sentence]
                current_title, current_summary = self.get_title_summary(sentence)
            else:
                if self.should_merge(sentence, current_title, current_summary):
                    current_chunk.append(sentence)
                    chunk_text = " ".join(current_chunk)
                    current_title, current_summary = self.get_title_summary(chunk_text)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_title, current_summary = self.get_title_summary(sentence)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_strategy_name(self) -> str:
        return "LLM Chunking"

class PlanningAgent:
    """Agent 1: Lập kế hoạch chunking"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.planning_prompt = PromptTemplate.from_template("""
        Bạn là Planning Agent trong hệ thống Agentic Chunking.
        
        Nhiệm vụ: Phân tích văn bản và đề xuất chiến lược chunking phù hợp nhất.
        
        Văn bản cần phân tích:
        "{text}"
        
        Các chiến lược có sẵn:
        1. Character Splitting - Chia theo số ký tự cố định
        2. Recursive Character Text Splitting - Chia theo thứ tự ưu tiên: heading -> đoạn -> dòng -> từ
        3. Semantic Chunking - Chia theo ngữ nghĩa bằng embedding similarity
        4. LLM Chunking - Chia bằng LLM để đảm bảo tính liên quan về chủ đề
        
        Hãy phân tích:
        - Độ dài văn bản
        - Cấu trúc văn bản (có heading, đoạn văn rõ ràng không?)
        - Tính đa dạng chủ đề
        - Độ phức tạp ngữ nghĩa
        
        Đề xuất chiến lược tốt nhất và giải thích lý do.
        Trả lời theo format:
        STRATEGY: <tên chiến lược>
        REASON: <lý do chi tiết>
        PARAMETERS: <các tham số đề xuất nếu có>
        """)
    
    def plan(self, text: str) -> Dict[str, Any]:
        """Lập kế hoạch chunking cho văn bản"""
        prompt = self.planning_prompt.format(text=text[:1000])  # Giới hạn để tiết kiệm token
        response = self.llm.invoke(prompt).content
        
        # Parse response
        lines = response.strip().split('\n')
        strategy = ""
        reason = ""
        parameters = ""
        
        for line in lines:
            if line.startswith("STRATEGY:"):
                strategy = line.replace("STRATEGY:", "").strip()
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
            elif line.startswith("PARAMETERS:"):
                parameters = line.replace("PARAMETERS:", "").strip()
        
        return {
            "strategy": strategy,
            "reason": reason,
            "parameters": parameters
        }

class ExecutionAgent:
    """Agent 2: Thực thi chunking"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.strategies = {
            "Character Splitting": CharacterSplittingStrategy(),
            "Recursive Character Text Splitting": RecursiveCharacterSplittingStrategy(),
            "Semantic Chunking": SemanticChunkingStrategy(openai_api_key),
            "LLM Chunking": LLMChunkingStrategy(openai_api_key)
        }
    
    def execute(self, text: str, plan: Dict[str, Any]) -> List[str]:
        """Thực thi chunking theo kế hoạch"""
        strategy_name = plan["strategy"]
        
        if strategy_name not in self.strategies:
            # Fallback về Recursive nếu không tìm thấy strategy
            strategy_name = "Recursive Character Text Splitting"
        
        strategy = self.strategies[strategy_name]
        chunks = strategy.chunk_text(text)
        
        return chunks

class ValidationAgent:
    """Agent 3: Đánh giá chất lượng chunks"""
    
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.validation_prompt = PromptTemplate.from_template("""
        Bạn là Validation Agent trong hệ thống Agentic Chunking.
        
        Nhiệm vụ: Đánh giá chất lượng các chunks đã tạo.
        
        Văn bản gốc:
        "{original_text}"
        
        Các chunks đã tạo:
        {chunks}
        
        Hãy đánh giá theo các tiêu chí:
        1. Tính toàn vẹn: Có mất thông tin không?
        2. Tính liên quan: Mỗi chunk có nội dung liên quan với nhau không?
        3. Độ dài phù hợp: Chunks có quá dài hoặc quá ngắn không?
        4. Tính độc lập: Mỗi chunk có thể hiểu được một cách độc lập không?
        5. Tính cân bằng: Các chunks có kích thước tương đối cân bằng không?
        
        Cho điểm từ 1-10 và đề xuất cải thiện nếu điểm < 7.
        
        Trả lời theo format:
        SCORE: <điểm số>
        EVALUATION: <đánh giá chi tiết>
        SUGGESTIONS: <đề xuất cải thiện nếu cần>
        """)
    
    def validate(self, original_text: str, chunks: List[str]) -> Dict[str, Any]:
        """Đánh giá chất lượng chunks"""
        chunks_text = "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])
        
        prompt = self.validation_prompt.format(
            original_text=original_text[:1000],  # Giới hạn để tiết kiệm token
            chunks=chunks_text[:1500]
        )
        
        response = self.llm.invoke(prompt).content
        
        # Parse response
        lines = response.strip().split('\n')
        score = 0
        evaluation = ""
        suggestions = ""
        
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except:
                    score = 5
            elif line.startswith("EVALUATION:"):
                evaluation = line.replace("EVALUATION:", "").strip()
            elif line.startswith("SUGGESTIONS:"):
                suggestions = line.replace("SUGGESTIONS:", "").strip()
        
        return {
            "score": score,
            "evaluation": evaluation,
            "suggestions": suggestions,
            "is_good": score >= 7
        }

class AgenticChunking:
    """Hệ thống Agentic Chunking chính"""
    
    def __init__(self, openai_api_key: str):
        self.planning_agent = PlanningAgent(openai_api_key)
        self.execution_agent = ExecutionAgent(openai_api_key)
        self.validation_agent = ValidationAgent(openai_api_key)
        self.max_iterations = 3  # Giới hạn số lần thử lại
    
    def chunk_text(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        """Thực hiện agentic chunking với feedback loop"""
        
        iteration = 0
        best_result = None
        best_score = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n🔄 Iteration {iteration}")
                print("=" * 50)
            
            # Agent 1: Planning
            if verbose:
                print("🤖 Planning Agent đang phân tích...")
            plan = self.planning_agent.plan(text)
            
            if verbose:
                print(f"📋 Chiến lược đề xuất: {plan['strategy']}")
                print(f"💡 Lý do: {plan['reason']}")
            
            # Agent 2: Execution
            if verbose:
                print("⚡ Execution Agent đang thực thi...")
            chunks = self.execution_agent.execute(text, plan)
            
            if verbose:
                print(f"📦 Đã tạo {len(chunks)} chunks")
            
            # Agent 3: Validation
            if verbose:
                print("✅ Validation Agent đang đánh giá...")
            validation = self.validation_agent.validate(text, chunks)
            
            if verbose:
                print(f"🎯 Điểm số: {validation['score']}/10")
                print(f"📝 Đánh giá: {validation['evaluation']}")
                if validation['suggestions']:
                    print(f"💭 Đề xuất: {validation['suggestions']}")
            
            # Lưu kết quả tốt nhất
            if validation['score'] > best_score:
                best_score = validation['score']
                best_result = {
                    "chunks": chunks,
                    "plan": plan,
                    "validation": validation,
                    "iteration": iteration
                }
            
            # Nếu đạt chất lượng tốt, dừng lại
            if validation['is_good']:
                if verbose:
                    print("✨ Chất lượng đạt yêu cầu!")
                break
            
            if verbose:
                print("❌ Chất lượng chưa đạt, thử lại...")
        
        if verbose:
            print(f"\n🏆 Kết quả tốt nhất từ iteration {best_result['iteration']} với điểm {best_score}/10")
        
        return best_result

# Ví dụ sử dụng
def demo_agentic_chunking():
    # Cần thay thế bằng API key thật
    OPENAI_API_KEY = userdata.get("open_ai_key")
    
    # Văn bản mẫu
    text = """Python là ngôn ngữ lập trình phổ biến. Nó được sử dụng rộng rãi trong phân tích dữ liệu. Mèo là động vật có vú. Chúng thường được nuôi làm thú cưng. Chó cũng là thú cưng phổ biến."""
    
    # Khởi tạo hệ thống
    agentic_chunker = AgenticChunking(OPENAI_API_KEY)
    
    # Thực hiện chunking
    result = agentic_chunker.chunk_text(text, verbose=True)
    
    # In kết quả cuối cùng
    print("\n" + "="*60)
    print("📊 KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    
    for i, chunk in enumerate(result["chunks"]):
        print(f"\n🔹 Chunk {i+1}:")
        print(f"{chunk}")
        print("-" * 40)
    
    print(f"\n✨ Chiến lược đã sử dụng: {result['plan']['strategy']}")
    print(f"🎯 Điểm chất lượng: {result['validation']['score']}/10")

demo_agentic_chunking()
