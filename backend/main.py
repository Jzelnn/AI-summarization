"""
FastAPI Backend with AI Model Selection, Evaluation Metrics, and Multilingual Support
Supports Gemini and HuggingFace T5-Small with comparison and quality metrics
FIXED: Non-English NER issue and httpx timeout issue
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import io
import os
import time

# Import modules
from input.speech_to_text import speech_to_text
from utils.file_reader import read_txt, read_docx, read_pdf

# FIXED: Import the multilingual-aware entity extraction
from nlp.pipeline import extract_entities_multilingual, detect_topic

# Import evaluation and multilingual support
try:
    from nlp.evaluation_metrics import SummarizationEvaluator
    EVALUATION_AVAILABLE = True
    print("✅ Evaluation metrics enabled")
except Exception as e:
    EVALUATION_AVAILABLE = False
    print(f"⚠️ Evaluation metrics not available: {e}")

try:
    from nlp.multilingual import MultilingualProcessor
    MULTILINGUAL_AVAILABLE = True
    print("✅ Multilingual support enabled")
except Exception as e:
    MULTILINGUAL_AVAILABLE = False
    print(f"⚠️ Multilingual support not available: {e}")

# Try to import both AI backends
GEMINI_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

try:
    from nlp.ai_summarizer_gemini import GeminiSummarizer
    gemini_summarizer = GeminiSummarizer()
    GEMINI_AVAILABLE = True
    print("✅ Gemini AI enabled")
except Exception as e:
    print(f"⚠️ Gemini not available: {e}")

try:
    from nlp.ai_summarizer_huggingface import AISummarizer
    # Use T5-small for faster performance
    hf_summarizer = AISummarizer("t5-small")
    HUGGINGFACE_AVAILABLE = True
    print("✅ HuggingFace AI enabled")
except Exception as e:
    print(f"⚠️ HuggingFace not available: {e}")

# Fallback to basic if neither available
if not GEMINI_AVAILABLE and not HUGGINGFACE_AVAILABLE:
    from nlp.summarizer import summarize as basic_summarize
    print("⚠️ Using basic summarization (no AI)")

# Initialize evaluator and multilingual processor
evaluator = SummarizationEvaluator() if EVALUATION_AVAILABLE else None
ml_processor = MultilingualProcessor() if MULTILINGUAL_AVAILABLE else None

app = FastAPI(
    title="AI-Powered NLP Summarizer with Evaluation & Multilingual Support",
    description="Compare Gemini vs HuggingFace with quality metrics + multilingual support",
    version="3.0.1-fixed"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def summarize_with_model(text: str, model: str, level: str, topic: str = None) -> tuple:
    """
    Summarize text using specified model
    
    Returns:
        (summary, processing_time)
    """
    start_time = time.time()
    
    if model == "gemini" and GEMINI_AVAILABLE:
        summary = gemini_summarizer.summarize(text, level=level, topic=topic)
    elif model == "huggingface" and HUGGINGFACE_AVAILABLE:
        summary = hf_summarizer.summarize(text, level=level, topic=topic)
    else:
        # Fallback to basic
        summary = basic_summarize(text, level=level, topic=topic)
    
    processing_time = time.time() - start_time
    return summary, processing_time


def evaluate_summary(original: str, summary: str) -> Dict:
    """Evaluate summary quality with metrics"""
    if not evaluator:
        return {}
    
    try:
        metrics = evaluator.evaluate(original, summary, include_all_metrics=True)
        return {
            'rouge1_f1': metrics.get('rouge1_f1', 0),
            'rouge2_f1': metrics.get('rouge2_f1', 0),
            'rougeL_f1': metrics.get('rougeL_f1', 0),
            'bleu': metrics.get('bleu', 0),
            'content_f1': metrics.get('content_f1', 0),
            'content_precision': metrics.get('content_precision', 0),
            'content_recall': metrics.get('content_recall', 0),
            'entity_preservation': metrics.get('entity_preservation', 0),
            'compression_ratio': metrics.get('compression_ratio', 0),
        }
    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "AI-Powered NLP Summarizer API with Evaluation & Multilingual (FIXED)",
        "version": "3.0.1-fixed",
        "fixes": [
            "Non-English NER now translates to English first",
            "HuggingFace T5-small httpx timeout issue resolved"
        ],
        "features": {
            "gemini": GEMINI_AVAILABLE,
            "huggingface": HUGGINGFACE_AVAILABLE,
            "evaluation_metrics": EVALUATION_AVAILABLE,
            "multilingual": MULTILINGUAL_AVAILABLE
        },
        "endpoints": {
            "upload_file": "/api/upload",
            "speech_to_text": "/api/speech-to-text",
            "analyze_text": "/api/analyze",
            "multilingual_analyze": "/api/multilingual-analyze",
            "compare_models": "/api/compare",
            "detect_language": "/api/detect-language",
            "translate": "/api/translate"
        }
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    level: str = Form("medium"),
    model: str = Form("gemini"),
    output_lang: str = Form(None)
):
    """
    Upload and process a file with AI model selection
    FIXED: NER now works properly with non-English text
    """
    try:
        # Read file content
        file_extension = file.filename.split(".")[-1].lower()
        content = await file.read()
        file_like = io.BytesIO(content)
        
        if file_extension == "txt":
            text = read_txt(file_like)
        elif file_extension == "docx":
            text = read_docx(file_like)
        elif file_extension == "pdf":
            text = read_pdf(file_like)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Detect language first
        detected_lang = ml_processor.detect_language(text) if ml_processor else 'en'
        
        # Handle multilingual if requested
        if output_lang and ml_processor:
            result = ml_processor.multilingual_summarize(
                text,
                lambda t, **kwargs: summarize_with_model(t, model, level, kwargs.get('topic'))[0],
                output_lang=output_lang
            )
            text_for_processing = result['text_for_summary']
            detected_lang = result['detected_lang']
        else:
            text_for_processing = text
        
        # Detect topic
        topic = detect_topic(text_for_processing)
        
        # Generate summary with selected model
        summary, proc_time = summarize_with_model(text_for_processing, model, level, topic)
        
        # Translate summary if needed
        if output_lang and ml_processor and output_lang != 'en':
            summary = ml_processor.translate_text(summary, target_lang=output_lang)
        
        # Evaluate summary
        metrics = evaluate_summary(text_for_processing, summary)
        
        # FIXED: Extract entities with multilingual support
        entities = extract_entities_multilingual(text, detected_lang)
        
        return {
            "status": "success",
            "filename": file.filename,
            "model_used": model,
            "detected_language": detected_lang,
            "output_language": output_lang or detected_lang,
            "original_text": text,
            "topic": topic,
            "summary": summary,
            "entities": entities,
            "level": level,
            "processing_time": f"{proc_time:.2f}s",
            "evaluation_metrics": metrics,
            "comparison": {
                "original_length": len(text),
                "original_words": len(text.split()),
                "summary_length": len(summary),
                "summary_words": len(summary.split()),
                "compression_ratio": f"{(len(summary) / len(text) * 100):.1f}%"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/speech-to-text")
async def speech_to_text_endpoint(
    audio: UploadFile = File(...),
    level: str = Form("medium"),
    model: str = Form("gemini"),
    output_lang: str = Form(None)
):
    """
    Convert speech to text and analyze
    FIXED: NER now works with non-English audio
    """
    try:
        audio_bytes = await audio.read()
        
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Convert speech to text
        text = speech_to_text(audio_bytes)
        
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No speech detected in audio")
        
        # Detect language
        detected_lang = ml_processor.detect_language(text) if ml_processor else 'en'
        
        # Handle multilingual
        if output_lang and ml_processor:
            result = ml_processor.multilingual_summarize(
                text,
                lambda t, **kwargs: summarize_with_model(t, model, level, kwargs.get('topic'))[0],
                output_lang=output_lang
            )
            text_for_processing = result['text_for_summary']
            detected_lang = result['detected_lang']
        else:
            text_for_processing = text
        
        # Detect topic
        topic = detect_topic(text_for_processing)
        
        # Generate summary
        summary, proc_time = summarize_with_model(text_for_processing, model, level, topic)
        
        # Translate if needed
        if output_lang and ml_processor and output_lang != 'en':
            summary = ml_processor.translate_text(summary, target_lang=output_lang)
        
        # Evaluate
        metrics = evaluate_summary(text_for_processing, summary)
        
        # FIXED: Extract entities with multilingual support
        entities = extract_entities_multilingual(text, detected_lang)
        
        return {
            "status": "success",
            "model_used": model,
            "detected_language": detected_lang,
            "output_language": output_lang or detected_lang,
            "transcribed_text": text,
            "topic": topic,
            "summary": summary,
            "entities": entities,
            "level": level,
            "processing_time": f"{proc_time:.2f}s",
            "evaluation_metrics": metrics
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/analyze")
async def analyze_text(
    text: str = Form(...),
    level: str = Form("medium"),
    model: str = Form("gemini"),
    output_lang: str = Form(None)
):
    """
    Analyze plain text with AI model selection
    FIXED: NER now works with non-English text
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Detect language
        detected_lang = ml_processor.detect_language(text) if ml_processor else 'en'
        
        # Handle multilingual
        if output_lang and ml_processor:
            result = ml_processor.multilingual_summarize(
                text,
                lambda t, **kwargs: summarize_with_model(t, model, level, kwargs.get('topic'))[0],
                output_lang=output_lang
            )
            text_for_processing = result['text_for_summary']
            detected_lang = result['detected_lang']
        else:
            text_for_processing = text
        
        # Detect topic
        topic = detect_topic(text_for_processing)
        
        # Generate summary
        summary, proc_time = summarize_with_model(text_for_processing, model, level, topic)
        
        # Translate if needed
        if output_lang and ml_processor and output_lang != 'en':
            summary = ml_processor.translate_text(summary, target_lang=output_lang)
        
        # Evaluate
        metrics = evaluate_summary(text_for_processing, summary)
        
        # FIXED: Extract entities with multilingual support
        entities = extract_entities_multilingual(text, detected_lang)
        
        return {
            "status": "success",
            "model_used": model,
            "detected_language": detected_lang,
            "output_language": output_lang or detected_lang,
            "original_text": text,
            "topic": topic,
            "summary": summary,
            "entities": entities,
            "level": level,
            "processing_time": f"{proc_time:.2f}s",
            "evaluation_metrics": metrics,
            "comparison": {
                "original_length": len(text),
                "original_words": len(text.split()),
                "summary_length": len(summary),
                "summary_words": len(summary.split()),
                "compression_ratio": f"{(len(summary) / len(text) * 100):.1f}%"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/compare")
async def compare_models(
    text: str = Form(...),
    level: str = Form("medium")
):
    """
    Compare both models side by side with evaluation metrics
    """
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Detect topic
        topic = detect_topic(text)
        
        results = {}
        
        # Generate summaries with both models
        if GEMINI_AVAILABLE:
            summary_gemini, time_gemini = summarize_with_model(text, "gemini", level, topic)
            metrics_gemini = evaluate_summary(text, summary_gemini)
            results['gemini'] = {
                'summary': summary_gemini,
                'processing_time': f"{time_gemini:.2f}s",
                'metrics': metrics_gemini,
                'word_count': len(summary_gemini.split())
            }
        
        if HUGGINGFACE_AVAILABLE:
            summary_hf, time_hf = summarize_with_model(text, "huggingface", level, topic)
            metrics_hf = evaluate_summary(text, summary_hf)
            results['huggingface'] = {
                'summary': summary_hf,
                'processing_time': f"{time_hf:.2f}s",
                'metrics': metrics_hf,
                'word_count': len(summary_hf.split())
            }
        
        # Determine winner
        winner = None
        if results:
            best_score = -1
            for model_name, data in results.items():
                score = data['metrics'].get('rouge1_f1', 0)
                if score > best_score:
                    best_score = score
                    winner = model_name
        
        return {
            "status": "success",
            "original_text": text,
            "topic": topic,
            "level": level,
            "results": results,
            "winner": winner,
            "comparison": {
                "original_words": len(text.split()),
                "models_compared": len(results)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/detect-language")
async def detect_language(text: str = Form(...)):
    """Detect the language of input text"""
    if not ml_processor:
        raise HTTPException(status_code=503, detail="Multilingual support not available")
    
    try:
        detected = ml_processor.detect_language(text)
        info = ml_processor.get_language_info(detected or 'en')
        
        return {
            "status": "success",
            "detected_language": detected,
            "language_name": info['name'],
            "language_emoji": info['emoji']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/translate")
async def translate_text_api(
    text: str = Form(...),
    target_lang: str = Form(...),
    source_lang: str = Form("auto")
):
    """Translate text to target language"""
    if not ml_processor:
        raise HTTPException(status_code=503, detail="Multilingual support not available")
    
    try:
        translated = ml_processor.translate_text(text, target_lang, source_lang)
        
        return {
            "status": "success",
            "original_text": text,
            "translated_text": translated,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    if not ml_processor:
        raise HTTPException(status_code=503, detail="Multilingual support not available")
    
    return {
        "languages": ml_processor.list_supported_languages()
    }


@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models"""
    return {
        "models": [
            {
                "id": "gemini",
                "name": "Gemini 2.5 Flash",
                "description": "Google's fast AI model",
                "available": GEMINI_AVAILABLE,
                "speed": "Very Fast",
                "quality": "Excellent"
            },
            {
                "id": "huggingface",
                "name": "T5-Small (Fixed)",
                "description": "Lightweight HuggingFace model (httpx issue fixed)",
                "available": HUGGINGFACE_AVAILABLE,
                "speed": "Fast",
                "quality": "Good"
            }
        ]
    }


@app.get("/api/config")
async def get_configuration():
    """Get current configuration"""
    return {
        "version": "3.0.1-fixed",
        "gemini_available": GEMINI_AVAILABLE,
        "huggingface_available": HUGGINGFACE_AVAILABLE,
        "evaluation_available": EVALUATION_AVAILABLE,
        "multilingual_available": MULTILINGUAL_AVAILABLE,
        "comparison_enabled": GEMINI_AVAILABLE and HUGGINGFACE_AVAILABLE,
        "fixes": [
            "Non-English NER: Text is translated to English before entity extraction",
            "HuggingFace T5: httpx.TimeoutException issue resolved"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)