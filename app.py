"""
app.py — Local Flask Web Server for JMI Commerce
Serves an executive light-mode website connected to models.py.
Exposes real-time endpoints for machine learning inference and training accuracy scores.
"""

import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

from ann_model import predict_retail_sales
from models import JMICommerceModels

app = Flask(__name__)
ANN_MODEL_PATH = Path(__file__).with_name('ann_model.pkl')

# Initialize ML Engine using Retail_sales.csv
print("Initializing JMI Commerce ML Engine...")
ml_engine = JMICommerceModels(csv_path='Retail_sales.csv')
print("ML Engine loaded successfully!")
print("Model Accuracy Scores:", ml_engine.get_model_accuracies())

ann_artifact = None
if ANN_MODEL_PATH.is_file():
    ann_artifact = joblib.load(ANN_MODEL_PATH)
    print(f"ANN demand model loaded from {ANN_MODEL_PATH}")
else:
    print(
        "ANN demand model is not trained yet. "
        "Run train_ann_model.py to enable /api/ann/predict."
    )


def _ann_number(data, key, minimum, maximum=None):
    value = data.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number.") from None
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{key} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{key} must be no more than {maximum}.")
    return number


@app.route('/')
def home():
    return render_template(
        'index.html',
        ann_model_available=ann_artifact is not None
    )

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns dataset metrics and trained model accuracy percentages."""
    return jsonify({
        'total_records': f"{len(ml_engine.df):,}",
        'unique_products': f"{ml_engine.df['ITEM CODE'].nunique():,}",
        'suppliers': f"{ml_engine.df['SUPPLIER'].nunique():,}",
        'avg_sales': f"${ml_engine.df['RETAIL SALES'].mean():.2f}",
        'accuracies': ml_engine.get_model_accuracies()
    })

@app.route('/api/accuracies', methods=['GET'])
def get_accuracies():
    """Endpoint returning model training accuracy percentages."""
    return jsonify(ml_engine.get_model_accuracies())

@app.route('/api/predict_sales', methods=['POST'])
def predict_sales():
    data = request.get_json()
    sales = ml_engine.predict_sales(
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'predicted_sales': round(sales, 2),
        'model_accuracy': f"{ml_engine.lr_accuracy}%"
    })

@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    data = request.get_json()
    pred, prob = ml_engine.predict_slow_moving(
        float(data.get('retail_sales', 0)),
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'is_slow_moving': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.dt_accuracy}%"
    })

@app.route('/api/predict_anomaly', methods=['POST'])
def predict_anomaly():
    data = request.get_json()
    pred, prob = ml_engine.predict_anomaly(
        float(data.get('retail_sales', 0)),
        float(data.get('retail_transfers', 0)),
        float(data.get('warehouse_sales', 0))
    )
    return jsonify({
        'is_anomaly': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.svm_accuracy}%"
    })

@app.route('/api/find_substitutes', methods=['POST'])
def find_substitutes():
    data = request.get_json()
    subs = ml_engine.find_substitutes(
        float(data.get('target_sales', 0)),
        float(data.get('target_transfers', 0)),
        float(data.get('target_warehouse', 0)),
        n_recommendations=3
    )
    return jsonify({
        'substitutes': subs,
        'model_accuracy': ml_engine.knn_accuracy
    })

@app.route('/api/classify_item', methods=['POST'])
def classify_item():
    data = request.get_json()
    desc = data.get('item_description', '')
    res = ml_engine.classify_item_type(desc)
    res['model_accuracy'] = f"{ml_engine.nb_accuracy}%"
    return jsonify(res)

@app.route('/api/predict_velocity', methods=['POST'])
def predict_velocity():
    data = request.get_json()
    pred, prob = ml_engine.predict_high_velocity(
        float(data.get('warehouse_sales', 0)),
        float(data.get('retail_transfers', 0)),
        int(data.get('month', 1))
    )
    return jsonify({
        'is_top_seller': pred,
        'probability': round(prob * 100, 1),
        'model_accuracy': f"{ml_engine.log_accuracy}%"
    })


@app.route('/api/ann/predict', methods=['POST'])
def predict_ann():
    if ann_artifact is None:
        return jsonify({
            'error': 'ANN model artifact is unavailable. Run train_ann_model.py first.'
        }), 503

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'A JSON object is required.'}), 400

    try:
        year = _ann_number(data, 'year', 2000, 2100)
        month = _ann_number(data, 'month', 1, 12)
        warehouse_sales = _ann_number(data, 'warehouse_sales', 0)
        retail_transfers = _ann_number(data, 'retail_transfers', 0)
        current_stock_units = _ann_number(data, 'current_stock_units', 0)
        unit_price = _ann_number(data, 'unit_price', 0)
        promo_uplift_pct = _ann_number(data, 'promo_uplift_pct', 0, 300)
        coverage_days = _ann_number(data, 'coverage_days', 1, 90)
    except ValueError as error:
        return jsonify({'error': str(error)}), 400

    if unit_price == 0:
        return jsonify({'error': 'unit_price must be greater than 0.'}), 400
    if any(not value.is_integer() for value in (year, month, coverage_days)):
        return jsonify({
            'error': 'year, month, and coverage_days must be whole numbers.'
        }), 400
    year, month, coverage_days = int(year), int(month), int(coverage_days)

    item_code = data.get('item_code')
    item_type = data.get('item_type')
    supplier = data.get('supplier')
    if not all(isinstance(value, str) and value.strip() for value in (
        item_code, item_type, supplier
    )):
        return jsonify({
            'error': 'item_code, item_type, and supplier must be non-empty strings.'
        }), 400

    features = pd.DataFrame([{
        'YEAR': year,
        'MONTH': month,
        'ITEM CODE': item_code.strip(),
        'ITEM TYPE': item_type.strip(),
        'SUPPLIER': supplier.strip(),
        'WAREHOUSE SALES': warehouse_sales,
        'RETAIL TRANSFERS': retail_transfers,
    }])
    predicted_monthly_sales = float(
        predict_retail_sales(ann_artifact['pipeline'], features)[0]
    )
    promo_factor = 1 + promo_uplift_pct / 100
    demand_units = (
        predicted_monthly_sales
        * promo_factor
        / unit_price
        * coverage_days
        / 30
    )
    stockout_risk_pct = (
        min(100.0, max(0.0, (demand_units - current_stock_units) / demand_units * 100))
        if demand_units > 0
        else 0.0
    )
    recommended_order_units = max(
        0,
        math.ceil(demand_units - current_stock_units),
    )

    return jsonify({
        'predicted_monthly_sales': round(predicted_monthly_sales, 2),
        'predicted_demand_units': round(demand_units, 2),
        'stockout_risk_pct': round(stockout_risk_pct, 1),
        'recommended_order_units': recommended_order_units,
        'coverage_days': coverage_days,
        'model_metrics': ann_artifact['metrics'],
        'risk_note': (
            'Stockout risk is an estimated uncovered-demand percentage, not a '
            'historically calibrated probability. Promotion uplift is a user-entered '
            'scenario because the dataset has no promotion history.'
        ),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)