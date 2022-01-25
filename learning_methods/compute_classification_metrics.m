function [accuracy, precision, recall, specificity, f1score] ...
    = compute_classification_metrics( y_test, true_y_test)

n_all       = numel(y_test);
niseq       = bsxfun(@eq, y_test, true_y_test);
niseq       = sum(niseq);
accuracy    = niseq/n_all*100;

tpositive   = true_y_test>0;
tpos        = sum(true_y_test(tpositive)==y_test(tpositive));
fpositivie  = y_test > 0;
fpos        = sum(true_y_test(fpositivie)<0);
precision   = tpos/(tpos+fpos)*100;

fnegative   = y_test <0;
fneg        = sum(true_y_test(fnegative)>0);

recall      = tpos / ( tpos + fneg ) * 100;

tnegative    = true_y_test < 0;
tneg        = sum(true_y_test(tnegative)==y_test(tnegative));
specificity = tneg / ( tneg + fpos)*100;

f1score     = 2*tpos/ ( 2*tpos + fpos + fneg)* 100;

end