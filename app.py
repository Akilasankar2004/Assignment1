from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model"
MAX_LENGTH = 128
TOP_K = 5
INTENT_NAMES = (
    "Card activation",
    "Opening an account for a child",
    "Apple Pay or Google Pay",
    "Supported ATMs",
    "Automatic top-ups",
    "Balance not updated after bank transfer",
    "Balance not updated after cheque or cash deposit",
    "Bank transfer beneficiary not allowed",
    "Canceling a transfer",
    "Card about to expire",
    "Where the card is accepted",
    "Card arrival",
    "Card delivery estimate",
    "Linking a card",
    "Card not working",
    "Card payment fee",
    "Unrecognized card payment",
    "Card payment exchange rate",
    "Card retained by ATM",
    "Cash withdrawal fee",
    "Unrecognized cash withdrawal",
    "Changing card PIN",
    "Compromised card",
    "Contactless payment not working",
    "Country availability",
    "Declined card payment",
    "Declined cash withdrawal",
    "Declined bank transfer",
    "Unrecognized direct debit",
    "Disposable card limits",
    "Changing personal details",
    "Currency exchange fee",
    "Exchange rate",
    "Exchanging currency in the app",
    "Extra charge on statement",
    "Failed transfer",
    "Supported currencies",
    "Disposable virtual card",
    "Getting a physical card",
    "Getting a spare card",
    "Getting a virtual card",
    "Lost or stolen card",
    "Lost or stolen phone",
    "Ordering a physical card",
    "Forgotten passcode",
    "Pending card payment",
    "Pending cash withdrawal",
    "Pending top-up",
    "Pending transfer",
    "Blocked PIN",
    "Receiving money",
    "Refund not showing up",
    "Requesting a refund",
    "Reverted card payment",
    "Supported cards and currencies",
    "Closing an account",
    "Bank transfer top-up fee",
    "Card top-up fee",
    "Cash or cheque top-up",
    "Failed top-up",
    "Top-up limits",
    "Reverted top-up",
    "Adding money by card",
    "Duplicate transaction charge",
    "Transfer fee",
    "Transferring money into an account",
    "Transfer not received by recipient",
    "Transfer taking too long",
    "Unable to verify identity",
    "Identity verification",
    "Source of funds verification",
    "Verify a top-up",
    "Virtual card not working",
    "Visa or Mastercard",
    "Why identity verification is needed",
    "Incorrect cash withdrawal amount",
    "Incorrect cash withdrawal exchange rate",
)


@st.cache_resource
def load_model():
    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"Model files were not found in {MODEL_PATH}.")

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_intents(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities, k=min(TOP_K, probabilities.numel())
    )
    return [
        (int(class_id), float(probability))
        for class_id, probability in zip(
            top_indices.cpu().tolist(), top_probabilities.cpu().tolist()
        )
    ]


st.set_page_config(
    page_title="Banking Intent Classifier",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 Banking Intent Classifier")
st.write(
    "Enter a banking support message to see the model's most likely intent classes."
)

with st.form("intent_prediction"):
    customer_message = st.text_area(
        "Customer message",
        placeholder="For example: I am still waiting for my new card.",
        height=140,
    )
    submitted = st.form_submit_button("Classify message", type="primary")

if submitted:
    if not customer_message.strip():
        st.warning("Enter a message before requesting a classification.")
    else:
        try:
            tokenizer, model, device = load_model()
        except OSError as error:
            st.error(f"Unable to load the saved model or tokenizer: {error}")
        else:
            predictions = predict_intents(
                customer_message.strip(), tokenizer, model, device
            )
            predicted_class, confidence = predictions[0]

            st.subheader("Prediction")
            st.write(f"This message appears to be about **{INTENT_NAMES[predicted_class]}**.")
            st.metric(
                "Prediction confidence",
                f"{confidence:.1%}",
            )

            probability_data = pd.DataFrame(
                {
                    "Intent": [
                        INTENT_NAMES[class_id] for class_id, _ in predictions
                    ],
                    "Confidence": [probability for _, probability in predictions],
                }
            ).set_index("Intent")
            st.caption("Other possible topics")
            st.bar_chart(probability_data, x_label="Intent", y_label="Confidence")

st.caption(
    "Predictions are suggestions and may not fully capture your request."
)
