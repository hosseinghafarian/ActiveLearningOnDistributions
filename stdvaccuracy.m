
accuracyt = accuracy
for i=1:18
    st = std(accuracyt(:, :, i), 0, 2);
    stdv(i, 1:50) = st'
end
    