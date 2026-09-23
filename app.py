"""Vercel-ready FastAPI frontend for the Health Condition Prediction System."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend import predict_health

app = FastAPI(title="Health Condition Prediction System")

class PredictionInput(BaseModel):
    sleep_duration: float = Field(gt=0)
    heart_rate: float = Field(gt=0)
    bmi: float = Field(gt=0)
    calorie_expenditure: float = Field(gt=0)
    step_count: float = Field(gt=0)
    exercise_duration: float = Field(gt=0)
    water_intake: float = Field(gt=0)
    diet_type: str
    stress_level: str
    sleep_quality: str
    physical_activity_level: str
    smoking_alcohol: str
    gender: str

HTML = r"""
<!doctype html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Health Condition Prediction</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f8fa;color:#123;max-width:1050px;margin:auto;padding:28px}
.card{background:white;padding:28px;border-radius:14px;box-shadow:0 8px 30px #0001}
h1{color:#174f78}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
label{font-weight:600}input,select{width:100%;padding:12px;margin-top:6px;box-sizing:border-box;border:1px solid #ccd8df;border-radius:8px}
button{width:100%;padding:14px;margin-top:20px;border:0;border-radius:8px;background:#1769aa;color:white;font-weight:bold;font-size:16px}
#result{margin-top:20px;padding:16px;border-radius:10px;display:none}
@media(max-width:700px){.grid{grid-template-columns:1fr}body{padding:14px}}
</style></head><body><div class="card">
<h1>Health Condition Prediction System</h1>
<p>Enter health, activity and lifestyle details to estimate the current condition.</p>
<form id="form"><div class="grid">
<label>Sleep duration (hours)<input name="sleep_duration" type="number" step="0.1" required></label>
<label>Heart rate (bpm)<input name="heart_rate" type="number" step="1" required></label>
<label>BMI<input name="bmi" type="number" step="0.1" required></label>
<label>Calorie expenditure<input name="calorie_expenditure" type="number" step="1" required></label>
<label>Daily step count<input name="step_count" type="number" step="1" required></label>
<label>Exercise duration (minutes)<input name="exercise_duration" type="number" step="1" required></label>
<label>Water intake (litres)<input name="water_intake" type="number" step="0.1" required></label>
<label>Diet type<select name="diet_type" required><option value="">Select</option><option>Balanced</option><option>Non-Veg</option><option>Veg</option></select></label>
<label>Stress level<select name="stress_level" required><option value="">Select</option><option>Low</option><option>Medium</option><option>High</option></select></label>
<label>Sleep quality<select name="sleep_quality" required><option value="">Select</option><option>Good</option><option>Average</option><option>Poor</option></select></label>
<label>Physical activity<select name="physical_activity_level" required><option value="">Select</option><option>Active</option><option>Moderate</option><option>Sedentary</option></select></label>
<label>Smoking / alcohol<select name="smoking_alcohol" required><option value="">Select</option><option>No</option><option>Occasional</option><option>Yes</option></select></label>
<label>Gender<select name="gender" required><option value="">Select</option><option>Female</option><option>Male</option><option>Other</option></select></label>
</div><button type="submit">Predict Health Condition</button></form>
<div id="result"></div></div>
<script>
const form=document.getElementById("form"),result=document.getElementById("result");
form.addEventListener("submit",async e=>{
 e.preventDefault();result.style.display="block";result.textContent="Analyzing...";
 const data=Object.fromEntries(new FormData(form).entries());
 for(const k of ["sleep_duration","heart_rate","bmi","calorie_expenditure","step_count","exercise_duration","water_intake"])data[k]=Number(data[k]);
 try{const r=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});
 const j=await r.json();if(!r.ok)throw new Error(j.detail||"Prediction failed");
 result.textContent="Prediction: "+j.prediction;
 }catch(err){result.textContent="Error: "+err.message}
});
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.post("/predict")
def predict(data: PredictionInput):
    return {"prediction": predict_health(data.model_dump())}
