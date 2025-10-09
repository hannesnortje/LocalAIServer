"""
Document Analyzer for Training Data Extraction
==============================================

This module analyzes methodology documents (like the 42 comprehensive analysis)
and extracts structured training data for QLoRA fine-tuning.

Key Features:
- Parse markdown documents and extract semantic sections
- Identify methodology patterns, principles, and processes
- Generate instruction-response pairs from document content
- Extract conversation examples and communication patterns
- Support multiple document types and structures

Philosophy Integration:
- Captures "42 = FOR TWO" collaborative intelligence principles
- Extracts systematic methodologies and process documentation
- Preserves communication style and philosophical approaches
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Structured content extracted from documents."""
    instruction: str
    context: str
    output: str
    category: str
    tags: List[str]
    difficulty: str = "intermediate"
    source_section: str = ""
    metadata: Dict[str, Any] = None

class DocumentAnalyzer:
    """
    Analyzes methodology documents and extracts training data.
    
    Supports multiple extraction patterns:
    - Philosophy and principle extraction
    - Process and methodology documentation
    - Conversation pattern analysis
    - Example scenario extraction
    """
    
    def __init__(self):
        self.section_patterns = {
            'philosophy': r'(?i)## (?:core )?philosophy|## (?:the )?fundamental|## universal|philosophy',
            'methodology': r'(?i)## (?:collaborative )?(?:intelligence )?(?:protocol|methodology|framework)',
            'process': r'(?i)## (?:implementation|steps?|workflow|procedure|protocol)',
            'examples': r'(?i)## (?:example|practical|implementation|case study)',
            'principles': r'(?i)## (?:principle|rule|guideline|standard)',
            'conversation': r'(?i)## (?:conversation|dialog|interaction|communication)',
            'crisis': r'(?i)## (?:crisis|prevention|emergency|problem)'
        }
        
        self.instruction_patterns = {
            'explain': "Explain the {concept}",
            'apply': "How do I apply {concept} in practice?",
            'implement': "Implement the {concept} methodology",
            'guide': "Guide me through {concept}",
            'example': "Provide an example of {concept}",
            'troubleshoot': "What should I do when {situation}?"
        }
        
    def analyze_document(self, file_path: str) -> List[ExtractedContent]:
        """
        Analyze a document and extract training data.
        
        Args:
            file_path: Path to the document to analyze
            
        Returns:
            List of extracted training content items
        """
        try:
            content = self._read_document(file_path)
            sections = self._parse_sections(content)
            extracted_items = []
            
            for section_type, sections_content in sections.items():
                items = self._extract_from_sections(section_type, sections_content)
                extracted_items.extend(items)
                
            logger.info(f"Extracted {len(extracted_items)} training items from {file_path}")
            return extracted_items
            
        except Exception as e:
            logger.error(f"Error analyzing document {file_path}: {e}")
            return []
    
    def _read_document(self, file_path: str) -> str:
        """Read document content."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _parse_sections(self, content: str) -> Dict[str, List[str]]:
        """Parse document into semantic sections."""
        sections = {category: [] for category in self.section_patterns.keys()}
        
        # Split content by main headers
        header_pattern = r'\n## (.+?)\n'
        parts = re.split(header_pattern, content)
        
        current_header = None
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Header
                current_header = part.strip()
            else:  # Content
                if current_header and part.strip():
                    category = self._categorize_section(current_header)
                    if category:
                        sections[category].append({
                            'header': current_header,
                            'content': part.strip()
                        })
        
        return sections
    
    def _categorize_section(self, header: str) -> Optional[str]:
        """Categorize section based on header content."""
        header_lower = header.lower()
        
        for category, pattern in self.section_patterns.items():
            if re.search(pattern, header_lower):
                return category
        
        # Default categorization based on keywords
        if any(word in header_lower for word in ['philosophy', 'principle', 'fundamental']):
            return 'philosophy'
        elif any(word in header_lower for word in ['process', 'step', 'protocol', 'workflow']):
            return 'process'
        elif any(word in header_lower for word in ['example', 'implementation', 'practical']):
            return 'examples'
        elif any(word in header_lower for word in ['methodology', 'framework', 'approach']):
            return 'methodology'
        
        return 'methodology'  # Default category
    
    def _extract_from_sections(self, section_type: str, sections: List[Dict]) -> List[ExtractedContent]:
        """Extract training data from specific section type."""
        extracted = []
        
        for section in sections:
            header = section['header']
            content = section['content']
            
            if section_type == 'philosophy':
                extracted.extend(self._extract_philosophy_training(header, content))
            elif section_type == 'methodology':
                extracted.extend(self._extract_methodology_training(header, content))
            elif section_type == 'process':
                extracted.extend(self._extract_process_training(header, content))
            elif section_type == 'examples':
                extracted.extend(self._extract_example_training(header, content))
            elif section_type == 'conversation':
                extracted.extend(self._extract_conversation_training(header, content))
            else:
                extracted.extend(self._extract_general_training(header, content, section_type))
        
        return extracted
    
    def _extract_philosophy_training(self, header: str, content: str) -> List[ExtractedContent]:
        """Extract training data from philosophy sections."""
        items = []
        
        # Extract key concepts
        concepts = self._extract_key_concepts(content)
        
        for concept in concepts:
            # Generate explanation instructions
            instruction = f"Explain the philosophy behind {concept}"
            output = self._extract_relevant_content(content, concept)
            
            if output:
                items.append(ExtractedContent(
                    instruction=instruction,
                    context="User needs to understand the foundational philosophy",
                    output=output,
                    category="philosophy",
                    tags=["philosophy", "42", "collaboration", concept.lower().replace(" ", "-")],
                    difficulty="foundational",
                    source_section=header
                ))
        
        # Extract principle-based training
        if "42" in content and "FOR TWO" in content:
            items.append(ExtractedContent(
                instruction="What does '42 = FOR TWO' mean in practice?",
                context="User learning collaborative intelligence methodology",
                output=self._extract_42_explanation(content),
                category="core_principle",
                tags=["42", "collaboration", "methodology", "core-principle"],
                difficulty="foundational",
                source_section=header
            ))
        
        return items
    
    def _extract_methodology_training(self, header: str, content: str) -> List[ExtractedContent]:
        """Extract training data from methodology sections."""
        items = []
        
        # Look for structured methodologies
        if "TRON Contribution" in content and "AI Contribution" in content:
            items.append(ExtractedContent(
                instruction="Explain the collaborative intelligence protocol",
                context="User needs systematic methodology guidance",
                output=self._extract_collaborative_protocol(content),
                category="methodology",
                tags=["collaboration", "protocol", "systematic-approach"],
                difficulty="intermediate",
                source_section=header
            ))
        
        # Extract process frameworks
        if "framework" in header.lower() or "protocol" in header.lower():
            items.append(ExtractedContent(
                instruction=f"How do I implement the {header.lower()}?",
                context="User needs implementation guidance",
                output=self._format_methodology_content(content),
                category="implementation",
                tags=["methodology", "framework", "implementation"],
                difficulty="intermediate",
                source_section=header
            ))
        
        return items
    
    def _extract_process_training(self, header: str, content: str) -> List[ExtractedContent]:
        """Extract training data from process sections."""
        items = []
        
        # Look for step-by-step processes
        steps = self._extract_numbered_steps(content)
        if steps:
            process_name = header.replace("## ", "").strip()
            
            items.append(ExtractedContent(
                instruction=f"Walk me through the {process_name} step by step",
                context="User needs systematic process guidance",
                output=self._format_steps(steps, process_name),
                category="process",
                tags=["process", "systematic", "steps"],
                difficulty="intermediate",
                source_section=header
            ))
        
        # Extract crisis prevention specifically
        if "crisis" in header.lower() or "prevention" in header.lower():
            items.append(ExtractedContent(
                instruction="What should I do when I'm overwhelmed by a complex problem?",
                context="User facing complex problem and feeling panicked",
                output=self._extract_crisis_protocol(content),
                category="crisis_management",
                tags=["42", "crisis-prevention", "collaboration", "systematic-approach"],
                difficulty="intermediate",
                source_section=header
            ))
        
        return items
    
    def _extract_example_training(self, header: str, content: str) -> List[ExtractedContent]:
        """Extract training data from example sections."""
        items = []
        
        # Look for real-world examples
        example_name = header.replace("## ", "").strip()
        
        items.append(ExtractedContent(
            instruction=f"Provide an example of {example_name.lower()}",
            context="User needs concrete implementation example",
            output=self._format_example_content(content),
            category="example",
            tags=["example", "practical", "implementation"],
            difficulty="intermediate",
            source_section=header
        ))
        
        return items
    
    def _extract_conversation_training(self, header: str, content: str) -> List[ExtractedContent]:
        """Extract conversation patterns for training."""
        items = []
        
        # Look for dialog patterns
        conversations = self._parse_conversations(content)
        
        for conv in conversations:
            items.append(ExtractedContent(
                instruction=conv['human'],
                context="User interaction following collaborative methodology",
                output=conv['assistant'],
                category="conversation",
                tags=["conversation", "interaction", "collaborative"],
                difficulty="intermediate",
                source_section=header
            ))
        
        return items
    
    def _extract_general_training(self, header: str, content: str, category: str) -> List[ExtractedContent]:
        """Extract general training data from any section."""
        items = []
        
        # Generate general instruction based on header
        instruction = f"Tell me about {header.replace('## ', '').strip().lower()}"
        
        items.append(ExtractedContent(
            instruction=instruction,
            context=f"User learning about {category}",
            output=self._clean_content(content),
            category=category,
            tags=[category, "methodology"],
            difficulty="intermediate",
            source_section=header
        ))
        
        return items
    
    # Helper methods for content extraction
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        concepts = []
        
        # Look for quoted concepts or emphasized terms
        quoted_matches = re.findall(r'"([^"]+)"', content)
        concepts.extend(quoted_matches)
        
        # Look for emphasized terms
        emphasized_matches = re.findall(r'\*\*([^*]+)\*\*', content)
        concepts.extend(emphasized_matches)
        
        # Look for specific patterns
        if "42 = FOR TWO" in content:
            concepts.append("42 = FOR TWO")
        if "collaborative intelligence" in content.lower():
            concepts.append("collaborative intelligence")
        
        return list(set(concepts))
    
    def _extract_42_explanation(self, content: str) -> str:
        """Extract 42 = FOR TWO explanation."""
        # Look for the core explanation
        pattern = r'(?i)42.*?FOR TWO.*?(?=\n\n|\n#|$)'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            explanation = match.group(0).strip()
            return self._format_explanation(explanation)
        
        return "42 = FOR TWO represents collaborative intelligence where complex problems require two perspectives working together."
    
    def _extract_collaborative_protocol(self, content: str) -> str:
        """Extract collaborative intelligence protocol details."""
        # Look for TRON and AI contribution sections
        tron_pattern = r'(?i)\*\*TRON Contribution.*?(?=\*\*AI Contribution|\n\n|$)'
        ai_pattern = r'(?i)\*\*AI Contribution.*?(?=\*\*|$)'
        
        tron_match = re.search(tron_pattern, content, re.DOTALL)
        ai_match = re.search(ai_pattern, content, re.DOTALL)
        
        protocol = "The collaborative intelligence protocol follows '42 = FOR TWO' philosophy:\n\n"
        
        if tron_match:
            protocol += tron_match.group(0).strip() + "\n\n"
        if ai_match:
            protocol += ai_match.group(0).strip() + "\n\n"
        
        protocol += "**Combined Intelligence:** 1 + 1 = 11 - exponentially superior results through strategic vision + systematic execution."
        
        return protocol
    
    def _extract_numbered_steps(self, content: str) -> List[str]:
        """Extract numbered steps from content."""
        step_pattern = r'(?:^|\n)\s*(?:\d+\.|\*|\-)\s*(.+?)(?=\n\s*(?:\d+\.|\*|\-)|$)'
        matches = re.findall(step_pattern, content, re.MULTILINE | re.DOTALL)
        
        return [step.strip() for step in matches if step.strip()]
    
    def _extract_crisis_protocol(self, content: str) -> str:
        """Extract crisis prevention protocol."""
        if "🛑" in content or "STOP" in content:
            # Extract the full protocol
            protocol_start = content.find("🛑")
            if protocol_start == -1:
                protocol_start = content.find("STOP")
            
            if protocol_start != -1:
                protocol = content[protocol_start:].split("\n\n")[0]
                return f"This is a perfect '42 = FOR TWO' moment! Let's apply the Crisis Prevention Protocol:\n\n{protocol}\n\nThe key is never working 'all one' when collaboration would produce better results."
        
        return "Apply the '42 = FOR TWO' principle: seek collaborative approach instead of working in isolation."
    
    def _parse_conversations(self, content: str) -> List[Dict[str, str]]:
        """Parse conversation examples from content."""
        conversations = []
        
        # Look for Human:/Assistant: patterns
        human_pattern = r'(?i)\*\*Human:\*\*\s*"([^"]+)"'
        assistant_pattern = r'(?i)\*\*(?:Assistant|TRON Response):\*\*\s*"([^"]+)"'
        
        human_matches = re.findall(human_pattern, content)
        assistant_matches = re.findall(assistant_pattern, content)
        
        for i, human_msg in enumerate(human_matches):
            if i < len(assistant_matches):
                conversations.append({
                    'human': human_msg,
                    'assistant': assistant_matches[i]
                })
        
        return conversations
    
    def _format_steps(self, steps: List[str], process_name: str) -> str:
        """Format steps into readable process."""
        formatted = f"Here's the {process_name} step by step:\n\n"
        
        for i, step in enumerate(steps, 1):
            formatted += f"{i}. {step}\n"
        
        return formatted.strip()
    
    def _format_methodology_content(self, content: str) -> str:
        """Format methodology content for training."""
        # Clean up and structure the content
        cleaned = self._clean_content(content)
        
        # Add implementation guidance
        if not cleaned.startswith("To implement"):
            cleaned = f"To implement this methodology:\n\n{cleaned}"
        
        return cleaned
    
    def _format_example_content(self, content: str) -> str:
        """Format example content for training."""
        cleaned = self._clean_content(content)
        
        if not cleaned.startswith("Here's a practical example"):
            cleaned = f"Here's a practical example:\n\n{cleaned}"
        
        return cleaned
    
    def _format_explanation(self, explanation: str) -> str:
        """Format explanation content."""
        # Remove extra whitespace and format nicely
        cleaned = re.sub(r'\n\s*\n', '\n\n', explanation)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _extract_relevant_content(self, content: str, concept: str) -> str:
        """Extract relevant content explaining a specific concept."""
        # Look for content around the concept
        lines = content.split('\n')
        relevant_lines = []
        
        for i, line in enumerate(lines):
            if concept.lower() in line.lower():
                # Get context around the match
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                relevant_lines.extend(lines[start:end])
                
        if relevant_lines:
            return '\n'.join(relevant_lines).strip()
        
        # Fallback: try to find any mention of key terms from the concept
        concept_words = concept.lower().split()
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in concept_words):
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                return '\n'.join(lines[start:end]).strip()
        
        # Final fallback: return first paragraph if nothing specific found
        paragraphs = content.split('\n\n')
        if paragraphs:
            return paragraphs[0][:500] + "..." if len(paragraphs[0]) > 500 else paragraphs[0]
        
        return content[:200] + "..." if len(content) > 200 else content
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content for training."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)
        
        # Remove metadata comments
        cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
        
        return cleaned.strip()