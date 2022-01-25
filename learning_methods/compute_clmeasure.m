function measure = compute_clmeasure(accuracyrep, precisionrep, recallrep, specificityrep, f1scorerep)

[measure.acc_avg, measure.acc_std]         = stat(accuracyrep);
[measure.prec_avg, measure.prec_std]       = stat(precisionrep);
[measure.recal_avg, measure.recal_std]     = stat(recallrep);
[measure.spec_avg, measure.spec_std]       = stat(specificityrep);
[measure.f1score_avg, measure.f1score_std] = stat(f1scorerep);

end