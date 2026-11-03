#!/bin/bash
# CPU-only watchdog (no GPU use): if the MAD-AD training job fails (e.g. out of memory because another process on the
# shared card grew), append a resumed training job (train_madad.py resumes from runs/unet_l_512/checkpoints/state.pt)
# and a fresh copy of the dev-sweep -> select -> maps -> finish chain depending on it. At most 3 retries.
#   screen -dmS uad_madad_watchdog bash sota/madad/watchdog.sh
cd /raid/home/CAMCA/hj880/perr
PYM="/raid/home/CAMCA/hj880/venvs/sota_madad/bin/python -I -B"
PYP="/raid/home/CAMCA/hj880/venvs/perr/bin/python -I -B"
TRAIN="$PYM sota/madad/train_madad.py --out sota/madad/runs/unet_l_512 --epochs 15 --warmup-epochs 0.75 --ckpt-every 2 --micro 32 --bf16 --workers 12"
log() { echo "$(date '+%m-%d %H:%M') $*" >> sota/madad/watchdog.log; }
cur=uad_madad_train
for try in 1 2 3; do
  until [ -f queue/rc/$cur ]; do sleep 60; done
  rc=$(cat queue/rc/$cur)
  if [ "$rc" = "0" ]; then log "$cur finished (rc 0)"; exit 0; fi
  next=uad_madad_train_r$try
  log "$cur failed (rc $rc); queueing $next and its chain"
  printf '24\t%s\t-\t%s\n' "$next" "$TRAIN" >> queue/jobs3.tsv
  printf '16\tuad_madad_dev_r%s\t%s\t%s sota/madad/infer_madad.py --run sota/madad/runs/unet_l_512 --keys sota/madad/keys_dev.txt --sweep --bs 8\n' "$try" "$next" "$PYM" >> queue/jobs3.tsv
  printf '0\tuad_madad_select_r%s\tuad_madad_dev_r%s\t%s sota/madad/select_madad.py\n' "$try" "$try" "$PYP" >> queue/jobs3.tsv
  printf '16\tuad_madad_maps_r%s\tuad_madad_select_r%s\t%s sota/madad/infer_madad.py --run sota/madad/runs/unet_l_512 --keys sota/madad/keys_dev.txt --chosen sota/madad/chosen.json --name madad --bs 8 && %s sota/madad/infer_madad.py --run sota/madad/runs/unet_l_512 --keys sota/madad/keys_test.txt --chosen sota/madad/chosen.json --name madad --bs 8\n' "$try" "$try" "$PYM" "$PYM" >> queue/jobs3.tsv
  printf '0\tuad_madad_finish_r%s\tuad_madad_maps_r%s\tbash sota/madad/finish.sh\n' "$try" "$try" >> queue/jobs3.tsv
  cur=$next
done
log "giving up after 3 retries"
