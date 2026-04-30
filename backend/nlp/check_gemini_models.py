"""
Check Available Gemini Models
This script lists all Gemini models available for your API key
"""

import google.generativeai as genai
import os

def check_gemini_models():
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("\nSet it with:")
        print("  PowerShell: $env:GEMINI_API_KEY = 'your-key'")
        print("  Linux/Mac: export GEMINI_API_KEY='your-key'")
        return
    
    print("=" * 80)
    print("CHECKING GEMINI MODELS")
    print("=" * 80)
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-5:]}")
    print("\n" + "-" * 80)
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        print("\n✅ Available models that support text generation:\n")
        
        available_models = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
                print(f"  ✓ {model.name}")
                print(f"    Description: {model.description[:80]}...")
                print(f"    Input token limit: {model.input_token_limit}")
                print(f"    Output token limit: {model.output_token_limit}")
                print()
        
        print("-" * 80)
        print(f"\nTotal models available: {len(available_models)}")
        
        # Recommend the best model
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR YOUR SUMMARIZATION APP")
        print("=" * 80)
        
        recommended = None
        if any('gemini-1.5-flash-latest' in m for m in available_models):
            recommended = 'gemini-1.5-flash-latest'
        elif any('gemini-1.5-flash' in m for m in available_models):
            recommended = 'gemini-1.5-flash'
        elif any('gemini-pro' in m for m in available_models):
            recommended = 'gemini-pro'
        
        if recommended:
            print(f"\n🎯 Recommended model: {recommended}")
            print(f"\nUpdate your ai_summarizer_gemini.py line 36 to:")
            print(f"   self.model = genai.GenerativeModel('{recommended}')")
        else:
            print("\n⚠️ No recommended models found. Try the first available model.")
        
        # Test the recommended model
        if recommended:
            print("\n" + "-" * 80)
            print("TESTING RECOMMENDED MODEL")
            print("-" * 80)
            
            try:
                model = genai.GenerativeModel(recommended)
                response = model.generate_content("Say 'Hello, I am working!'")
                print(f"\n✅ Test successful!")
                print(f"Response: {response.text}")
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
        
    except Exception as e:
        print(f"\n❌ Error connecting to Gemini API: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. API key doesn't have proper permissions")
        print("  3. Network connection issue")
        print("\nGet a new API key from: https://makersuite.google.com/app/apikey")

if __name__ == "__main__":
    check_gemini_models()