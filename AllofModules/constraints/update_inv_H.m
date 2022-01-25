function update_inv_H(k_coeff, Q_coeff)
global cnstData
    cnstData.H    = k_coeff*cnstData.K + Q_coeff*cnstData.Q;
    cnstData.Hinv = inv(cnstData.H);
end