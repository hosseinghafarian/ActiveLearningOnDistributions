function [Rind, Rvalue, b, constraint_to_instance_map, y_new, size_of_b, nz_max]                   = eq_AEC_diag_of_labeled_linearind_val(pre_const_to_inst_map, pre_y,c,elements_of_diag)
    % cConstraint= [cConstraint,diag(G_plus(initL,initL))==1-p(initL),...
    %                  diag(G_plus(setunlab,setunlab))==r,...
    % Constraint: diag(G_{ll,uu})==1-g, since g and a are removed it is
    % equivalent to g=p+q , (i.e,diag(G_{ll,uu})==1-(p+q))
    % this is equivalent to diag(G_{ll})+ p_l ==1
global cnstData
global cnstDefs
    nSDP  = cnstData.nSDP;
    initL = cnstData.initL(cnstData.initL>0);
    initLStart          = cnstData.initLStart; 
    initLStart_notnoisy = cnstData.initLStart_notnoisy;
    if initLStart_notnoisy 
       init_maybenoisy     = setdiff(initL, initLStart);
       init_notnoisy       = initLStart;
    else
       init_maybenoisy     = initL;
       init_notnoisy       = [];
    end
    nz_max = 2;
    % For those elements of initL which are notnoisy
    size_of_bS     = numel(init_notnoisy);
    if size_of_bS==0
       zerovecS       = [];
       onevecS        = [];
    else
       zerovecS       = zeros(1,size_of_bS);
       onevecS        = ones(1,size_of_bS);
    end
    rowsS          = initLStart; % it is also, equal cols
    valsS          = onevecS;
    y_newS         = zeros(size_of_bS,1);
    if ~isempty(pre_const_to_inst_map)
        ind_pre_initL = pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_DIAG_INITL_P & ismember(pre_const_to_inst_map(:,2),init_notnoisy); 
        pre_size_of_bS = sum(ind_pre_initL);
        y_newS(1:pre_size_of_bS)= pre_y(ind_pre_initL);
    end
    constraint_to_instance_mapS = [repmat(cnstDefs.SINGLE_INS,size_of_bS,1), init_notnoisy',repmat(cnstDefs.CSTR_ID_DIAG_INITL_P,size_of_bS,1)];
    [RindS , RvalueS] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, 1:size_of_bS, ...
                                rowsS, rowsS, valsS, 1:size_of_bS, init_notnoisy, zerovecS);
    bS     = elements_of_diag*onevecS;
    % For those elements of initL which is not been at the start
    size_of_bN      = numel(init_maybenoisy);
    if size_of_bN ==0
       onevecN = [];
    else
       onevecN         = ones(1,size_of_bN);
    end
    rowsN           = init_maybenoisy; % it is also, equal cols
    valsN           = onevecN;
    y_newN          = zeros(size_of_bN,1);
    if ~isempty(pre_const_to_inst_map)
        ind_pre_initL = pre_const_to_inst_map(:,3)== cnstDefs.CSTR_ID_DIAG_INITL_P & ismember(pre_const_to_inst_map(:,2),init_maybenoisy); 
        pre_size_of_b = sum(ind_pre_initL);
        y_newN(1:pre_size_of_b)= pre_y(ind_pre_initL);
    end
    constraint_to_instance_mapN = [repmat(cnstDefs.SINGLE_INS,size_of_bN,1), init_maybenoisy,repmat(cnstDefs.CSTR_ID_DIAG_INITL_P,size_of_bN,1)];
    [RindN , RvalueN] = convert_multiple_linear_indices(nz_max, nSDP, nSDP, 1:size_of_bN, ...
                                rowsN, rowsN, valsN, 1:size_of_bN, init_maybenoisy, onevecN);
    bN        = elements_of_diag*onevecN;
    % Result to return
    Rind     = [  RindN;  RindS];
    Rvalue   = [RvalueN;RvalueS];
    b        = [     bN,     bS];
    size_of_b= size_of_bN + size_of_bS;
    constraint_to_instance_map = [constraint_to_instance_mapN;constraint_to_instance_mapS];
    y_new    = [ y_newN; y_newS]; 
end