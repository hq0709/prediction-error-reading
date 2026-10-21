#!/bin/bash
# After the kept full-data verifier has re-scored every candidate set: concatenate its MIMIC shards, rerun the
# radiograph collectors and the appendix collector, refresh the curve figure and the appendix tables.
#     bash finish_cxr.sh            # the kept PER run from results/cxr_aug_choice.json
set -e
cd /raid/home/CAMCA/hj880/perr
PP=/raid/home/CAMCA/hj880/venvs/perr/bin/python
export OMP_NUM_THREADS=8
$PP choose_cxr_aug.py | tail -n 3
RUN=$($PP -c "from choose_cxr_aug import chosen; print(chosen('PER (ours)'))")
R=results/cxr_focal/$RUN
echo "kept full-data verifier: $RUN"
for f in $R/*_g_*.jsonl; do          # three-panel runs score <proposer>_g/<set>: link under the two-panel names
  [ -e "$f" ] || continue
  b=$(basename "$f"); ln -sf "$b" "$R/${b/_g_/_}"
done
for s in validate test; do
  n=$([ $s = validate ] && echo 1 || echo 2)
  files=$(for i in $(seq 0 $n); do echo $R/wmu_mimic_${s}_s$i.jsonl; done)
  for f in $files; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
  cat $files > $R/wmu_mimic_$s.jsonl
  echo "$R/wmu_mimic_$s.jsonl: $(wc -l < $R/wmu_mimic_$s.jsonl) rows"
done
$PP score_cxr_focal_mimic.py densenet121 $RUN > logs/finish_mimic.log 2>&1 && sed -n '/fixed rules/,$p' logs/finish_mimic.log
$PP collect_cxr_focal.py > logs/finish_focal.log 2>&1 && head -n 9 logs/finish_focal.log
$PP collect_cxr_curve.py > logs/finish_curve.log 2>&1 && sed -n '/^LROC/,/^$/p' logs/finish_curve.log
$PP cxr_paper_macros.py
$PP analysis_per.py > logs/finish_analysis.log 2>&1 && grep radiograph logs/finish_analysis.log
$PP collect_supp.py > logs/finish_supp.log 2>&1 && echo "collect_supp done"
cd /raid/home/CAMCA/hj880/perr-paper
python3 scripts/update_curves_json.py
eval "$(/raid/home/CAMCA/hj880/miniconda3/bin/conda shell.bash hook)" && conda activate base
python scripts/figures/fig3_candidates/plot_fig3_cvpr.py --variants 2 --no-qa > /dev/null 2>&1 && cp scripts/figures/fig3_candidates/fig3_cvpr_2.pdf figures/label_efficiency.pdf && echo "label-efficiency figure replotted"
python3 scripts/make_supp_tables.py | grep -v "^MATCH" | tail -n 30
