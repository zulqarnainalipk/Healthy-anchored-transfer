# Reruns every synthetic-benchmark module with harmonised class names (module 03b) into the standard result folders.
# The first-pass (unharmonised) results are copied to results\_first_pass_unharmonised before anything is overwritten.
# Run from the repository root:  powershell -File analysis\rerun_all_harmonised.ps1
$ErrorActionPreference = "Continue"
$env:HARMONISE = "1"
$F = "results\05_features_dinov2\remote\features_dinov2"
$L = "results\01_crop_disease_table\labelled_images.csv"
$P = "results\08_projection\projection_k16.npz"
$PP = "results\08b_projection_pca\projection_k16.npz"
$P4 = "results\08_projection\projection_k4.npz"
$P8 = "results\08_projection\projection_k8.npz"
function Log($m) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

Log "archiving first-pass results"
$arch = "results\_first_pass_unharmonised"
New-Item -ItemType Directory -Force $arch | Out-Null
foreach ($d in "17_benchmark_hatc","17_calibration_v2","18_agreement","20_align_baselines","26_linear_probes","30_certificate_baselines","31_labelled_saving","35_label_noise","36_cluster_bootstrap","37_equivalence","38_exact_test","41_d3_drift","42_d4_sequential","43_ablations","50_hatc_select","46_tables","46b_metrics_cost","46d_failure_cases","12_host_maps_brenier") {
    if ((Test-Path "results\$d") -and -not (Test-Path "$arch\$d")) { Copy-Item "results\$d" "$arch\$d" -Recurse }
}
if ((Test-Path "figures") -and -not (Test-Path "$arch\figures")) { Copy-Item "figures" "$arch\figures" -Recurse }

$groups = [ordered]@{ g1 = @("a","a2","b"); g2 = @("b2","c25","c50"); g3 = @("c100","d"); g4 = @("e","f") }

Log "step 1: 17, 18, 20 (main benchmark and agreement)"
python features\17_benchmark_hatc.py --features $F --labelled $L --projection $P --output results\17_benchmark_hatc --scale 2
python features\18_agreement_benchmark.py --features $F --labelled $L --projection $P --output results\18_agreement --scale 2
python features\20_align_baselines.py --features $F --labelled $L --projection $P --output results\20_align_baselines --scale 2

Log "step 2: 36, 36b, 37, 38 (statistics)"
python features\36_cluster_bootstrap.py --replicates results\20_align_baselines\replicate_results.csv --output results\36_cluster_bootstrap
python features\36b_bootstrap_coverage_power.py --replicates results\20_align_baselines\replicate_results.csv --output results\36b_bootstrap_coverage_power
python features\37_equivalence.py --intervals results\36_cluster_bootstrap\intervals.csv --output results\37_equivalence
python features\38_exact_test.py --replicates results\20_align_baselines\replicate_results.csv --output results\38_exact_test

Log "step 3: 26 probes and 30 certificate baselines (four parallel groups each)"
$procs = @()
foreach ($g in $groups.Keys) {
    $procs += Start-Process python -ArgumentList (@("features\26_linear_probes.py","--features",$F,"--labelled",$L,"--projection",$P,"--output","results\26_linear_probes\$g","--worlds") + $groups[$g]) -PassThru -NoNewWindow -RedirectStandardOutput "results\_rerun_26_$g.log" -RedirectStandardError "results\_rerun_26_$g.err"
    $procs += Start-Process python -ArgumentList (@("features\30_certificate_baselines.py","--features",$F,"--labelled",$L,"--projection",$P,"--output","results\30_certificate_baselines\$g","--scale","2","--worlds") + $groups[$g]) -PassThru -NoNewWindow -RedirectStandardOutput "results\_rerun_30_$g.log" -RedirectStandardError "results\_rerun_30_$g.err"
}
$procs | Wait-Process
Log "step 3b: merging group outputs (26 probe curve, 30 auroc)"
python analysis\merge_groups.py

Log "step 4: 31 labelled saving"
python features\31_labelled_saving.py --hatc results\20_align_baselines\replicate_results.csv --probes results\26_linear_probes\probe_curve.csv --output results\31_labelled_saving

Log "step 5: 43 ablations and 35 label noise"
python features\43_ablation_engine.py --features $F --labelled $L --projection $P --output results\43_ablations --tag sweep_n_k16 --healthy-n 10 25 50 100 200 400 --worlds a c50 e f --reps 100
python features\43_ablation_engine.py --features $F --labelled $L --projection $P4 --output results\43_ablations --tag dim_k4 --healthy-n 50 200 --worlds a c50 e f --reps 100
python features\43_ablation_engine.py --features $F --labelled $L --projection $P8 --output results\43_ablations --tag dim_k8 --healthy-n 50 200 --worlds a c50 e f --reps 100
python features\43_ablation_engine.py --features $F --labelled $L --projection $PP --output results\43_ablations --tag pca_k16 --healthy-n 50 200 --worlds a c50 e f --reps 100
foreach ($nz in 0, 0.05, 0.1) {
    python features\43_ablation_engine.py --features $F --labelled $L --projection $P --output results\35_label_noise --tag noise_$nz --healthy-n 200 --worlds a c50 e f --reps 100 --label-noise $nz
}

Log "step 6: 41, 42 (device D3, D4)"
python features\41_d3_drift_refit.py --features $F --labelled $L --projection $P --output results\41_d3_drift --scale 2
python features\42_d4_sequential.py --features $F --labelled $L --projection $P --output results\42_d4_sequential --scale 2

Log "step 7: 50 HATC-S (original seed and fresh seed), 12, 46b, 46d"
python features\50_hatc_select.py --features $F --labelled $L --projection $P --output results\50_hatc_select\seed2026 --scale 2 --world-seed 2026 --tag seed2026
python features\50_hatc_select.py --features $F --labelled $L --projection $P --output results\50_hatc_select\seed9002 --scale 2 --world-seed 9002 --tag seed9002
python features\12_host_maps_brenier.py --features $F --labelled $L --projection $P --output results\12_host_maps_brenier --tag k16 --healthy-n 200 400 --worlds a c50 e f --reps 58
python features\46b_metrics_cost.py --features $F --labelled $L --projection $P --output results\46b_metrics_cost --tag n200 --healthy-n 200 --worlds a c50 e f --reps 58
python features\46d_failure_cases.py --features $F --labelled $L --projection $P --output results\46d_failure_cases --tag n200 --healthy-n 200 --worlds a e --reps 58

Log "step 8: tables and figures"
python analysis\46_tables.py --results results --output results\46_tables
python analysis\47_figures.py --results results --output figures
Log "all done"
