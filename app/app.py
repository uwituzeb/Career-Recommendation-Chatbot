import gradio as gr
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

model_dir = './saved-models'
FALLBACK_RESPONSE = "Sorry, I am not able to answer that. Try asking a career-related question."

# Load model and tokenizer
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def generate_answer_with_confidence(model, tokenizer, question, threshold=-3.5, max_length=64):
    input_text = "question: " + question
    inputs = tokenizer(input_text, return_tensors="tf", padding="longest", truncation=True, max_length=128)

    # Generate answer
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Evaluate loss
    loss = model(input_ids=inputs["input_ids"], labels=output_ids).loss
    avg_log_prob = -loss.numpy()

   # Compare with threshold
    if avg_log_prob < threshold:
        return FALLBACK_RESPONSE
    else:
        return answer

# Build UI using gradio
with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown("## 🎯 Career Recommendation Bot")
    gr.Markdown(
        "👋 Welcome! I'm your personal career guide.\n"
        "Ask me questions about career pathways or job choices and get insightful recommendations tailored for you."
    )

    with gr.Column():
        question = gr.Textbox(
            lines=2,
            placeholder="Ask your career question... e.g., 'What skills do I need to be a data analyst?'",
            label="Question"
        )
        answer = gr.Textbox(label="Answer")

    with gr.Row():
        submit_button = gr.Button("Submit", variant="primary")
        clear_button = gr.Button("Clear")

    submit_button.click(
        fn=lambda q: generate_answer_with_confidence(model, tokenizer, q),
        inputs=question,
        outputs=answer
    )
    clear_button.click(lambda: ("", ""), None, [question, answer])

if __name__ == "__main__":
    demo.launch()