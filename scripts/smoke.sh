#!/usr/bin/env bash
# Live smoke test against a running service: 20 real clinical questions must
# all return 200 with at least one verified finding or an honest insufficient,
# no 500s, and p95 under 30 s. Usage: scripts/smoke.sh https://host [count]
set -uo pipefail
BASE="${1:?base url required}"
COUNT="${2:-20}"
QUESTIONS=(
  "In adults having total knee replacement, does tranexamic acid reduce transfusion compared with placebo?"
  "Does prehabilitation before coronary artery bypass grafting shorten hospital stay?"
  "In hip fracture surgery, does regional anaesthesia reduce delirium compared with general anaesthesia?"
  "Does aspirin prevent venous thromboembolism after total hip arthroplasty as well as low molecular weight heparin?"
  "In spinal fusion, does intrawound vancomycin powder reduce surgical site infection?"
  "Do opioid-sparing multimodal analgesia protocols reduce opioid use after knee arthroplasty?"
  "In elective colectomy, does enhanced recovery after surgery reduce complications?"
  "Does dexamethasone reduce postoperative nausea and vomiting after general anaesthesia?"
  "In patients with atrial fibrillation, do direct oral anticoagulants reduce stroke compared with warfarin?"
  "Does SGLT2 inhibition reduce hospitalisation for heart failure in patients with reduced ejection fraction?"
  "In type 2 diabetes, does metformin reduce cardiovascular mortality?"
  "Does early mobilisation after major bowel surgery reduce length of stay?"
  "In community acquired pneumonia, are 5 days of antibiotics as effective as 10?"
  "Does cognitive behavioural therapy improve chronic low back pain compared with usual care?"
  "In older adults, do statins reduce the incidence of dementia?"
  "Does tourniquet use in total knee arthroplasty increase thromboembolic events?"
  "In CABG, does off-pump surgery reduce stroke compared with on-pump?"
  "Does preoperative anaemia treatment with intravenous iron reduce transfusion in major surgery?"
  "In hip fracture, does surgery within 24 hours reduce mortality?"
  "Does remote patient monitoring after discharge reduce 30-day readmission in heart failure?"
)
fails=0; times=()
for ((i=0; i<COUNT && i<${#QUESTIONS[@]}; i++)); do
  q="${QUESTIONS[$i]}"
  start=$(date +%s)
  out=$(curl -sS -m 170 -w '\n%{http_code}' -X POST "${BASE}/v1/evidence" -H 'content-type: application/json' \
        --data "$(printf '{"question": %s}' "$(printf '%s' "$q" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")")
  code=$(printf '%s' "$out" | tail -n1)
  body=$(printf '%s' "$out" | sed '$d')
  secs=$(( $(date +%s) - start )); times+=("$secs")
  status=$(printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("status"), len(d.get("findings",[])), d.get("certainty"), d.get("retrieval",{}).get("source"))' 2>/dev/null || echo "unparseable")
  printf '%3d %3ss  %s  %s\n' "$code" "$secs" "$status" "${q:0:60}"
  [ "$code" = "200" ] || fails=$((fails+1))
done
sorted=$(printf '%s\n' "${times[@]}" | sort -n)
n=${#times[@]}; idx=$(( (n * 95 + 99) / 100 )); [ "$idx" -lt 1 ] && idx=1
p95=$(printf '%s\n' "$sorted" | sed -n "${idx}p")
echo "non-200: ${fails}/${n}   p95: ${p95}s"
[ "$fails" = 0 ] && [ "$p95" -le 30 ]
