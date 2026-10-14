
def calculate_signal(models, latest_close):
    model_names = [
        "Linear Regression",
        "Random Forest",
        "LSTM"
    ]

    buy_weight = 0.0
    sell_weight = 0.0

    for name in model_names:
        result = models[name]

        predicted_price = result["predicted_price"]
        accuracy = result["directional_accuracy"]

        # Use performance above 50% as voting weight
        weight = max(0.0, accuracy - 50.0)

        if predicted_price > latest_close:
            buy_weight += weight
        elif predicted_price < latest_close:
            sell_weight += weight

    total_weight = buy_weight + sell_weight

    if total_weight == 0:
        return {
            "signal": "UNRELIABLE",
            "score": 0.0,
            "message": "No model demonstrated directional accuracy above 50%."
        }

    if buy_weight > sell_weight:
        signal = "BUY"
    else:
        signal = "SELL"

    score = max(buy_weight, sell_weight) / total_weight * 100

    return {
        "signal": signal,
        "score": round(score, 2),
        "message": "Experimental model agreement score"
    }
