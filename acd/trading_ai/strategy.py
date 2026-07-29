# filename: strategy.py
import json
from kafka import KafkaConsumer

# Configurations
KAFKA_SERVER = 'localhost:9092'
PREDICTION_TOPIC = 'ionq_predictions'

# Strategy Parameters
CONFIDENCE_THRESHOLD = 0.005  # e.g., 0.5% minimum expected return to trigger trade
NO_TRADE_ZONE_LOWER = -0.002
NO_TRADE_ZONE_UPPER = 0.002
STOP_LOSS_PCT = 0.015         # 1.5% Stop Loss
TAKE_PROFIT_PCT = 0.03        # 3.0% Take Profit
BASE_POSITION_SIZE = 100      # Base shares/units to trade

def evaluate_strategy(prediction_data):
    pred_return = prediction_data.get('predicted_return', 0.0)
    current_close = prediction_data.get('current_close', 0.0)
    
    # Check No-Trade Zone
    if NO_TRADE_ZONE_LOWER <= pred_return <= NO_TRADE_ZONE_UPPER:
        print(f"Predicted Return {pred_return*100:.2f}% falls inside No-Trade Zone. ACTION: HOLD / NO ACTION")
        return
    
    signal = None
    if pred_return >= CONFIDENCE_THRESHOLD:
        signal = "BUY"
    elif pred_return <= -CONFIDENCE_THRESHOLD:
        signal = "SELL"
    else:
        print(f"Predicted Return {pred_return*100:.2f}% below confidence threshold. ACTION: HOLD")
        return
        
    # Calculate Risk Management Parameters
    if signal == "BUY":
        stop_loss_price = current_close * (1 - STOP_LOSS_PCT)
        take_profit_price = current_close * (1 + TAKE_PROFIT_PCT)
        print(f"Predicted Return +{pred_return*100:.2f}% ↓ BUY")
    else:
        stop_loss_price = current_close * (1 + STOP_LOSS_PCT)
        take_profit_price = current_close * (1 - TAKE_PROFIT_PCT)
        print(f"Predicted Return {pred_return*100:.2f}% ↓ SELL")
        
    # Position Sizing Logic (Scale size dynamically based on return magnitude)
    position_size = int(BASE_POSITION_SIZE * (abs(pred_return) / CONFIDENCE_THRESHOLD))
    
    print("-" * 50)
    print(f"EXECUTION ORDER GENERATED:")
    print(f"Signal: {signal}")
    print(f"Position Size: {position_size} units")
    print(f"Current Price: {current_close:.2f}")
    print(f"Stop Loss Price: {stop_loss_price:.2f}")
    print(f"Take Profit Price: {take_profit_price:.2f}")
    print("-" * 50)

def main():
    consumer = KafkaConsumer(
        PREDICTION_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    print("Strategy engine listening to predictions...")
    for message in consumer:
        prediction_payload = message.value
        evaluate_strategy(prediction_payload)

if __name__ == '__main__':
    main()