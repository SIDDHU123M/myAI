from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Initialize Flask app
app = Flask(__name__)

@app.route('/chat', methods=['GET'])
def chat():
    user_input = request.args.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Tokenize the input with padding and attention mask handling
        inputs = tokenizer.encode_plus(
            user_input, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512  # Ensure that inputs are not too long
        )

        # Generate response with improved diversity settings
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=150, 
            num_return_sequences=1, 
            temperature=0.8,  # Increase creativity
            no_repeat_ngram_size=2,  # Prevent repetitive n-grams
            top_p=0.9,  # Use nucleus sampling to limit token choices
            top_k=50    # Limit possible next tokens to top-k
        )
        
        # Decode the response and handle special tokens
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Error generating response: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
