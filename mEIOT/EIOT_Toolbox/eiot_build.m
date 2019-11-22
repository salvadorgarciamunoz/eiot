function  [eiot_obj] = eiot_build(dm,ck,num_si)
%function [eiot_obj] = eiotv2_build(Dm,Ck,num_si)
%
%Inputs:
% Dm       : Pre-processed matrix of mixture spectra
% Ck       : Known mass fractions of mixture spectra (Ck is kept for consistency with paper)
% num_si   : Number of non-chemical interference signatures to identify
%
%Outputs:
% eiot_obj.S_hat      : Estimation of apparent pure component spectra
% eiot_obj.S_I        : Non-chemical signatures
% eiot_obj.S_E        : Extended set of signatures (chemical and non)
% eiot_obj.E_ch       : Spectral error after chemical deflation
% eiot_obj.r_I        : Strength of non-chemical interferences
% eiot_obj.r_I_bounds : Upper and lower bounds for r_I
% eiot_obj.SR         : Spectral residual
% eiot_obj.SSR        : Squared sum of spectral residuals;
% eiot_obj.lambdas    : Singular values of Dm after removing chemical info.


S_hat  = pinv(ck'*ck)*ck'*dm;
Dm_hat = ck*S_hat;
if num_si>0
    E_ch   = dm-Dm_hat;
    [U,S,V]=svd(E_ch);
    S_I     = V(:,1:num_si);
    S_short = S(1:num_si,1:num_si);
    r_I     = U(:,1:num_si)*S_short;
    S_E = [S_hat',S_I]';
    SR  = dm-Dm_hat;
    SSR = sum(SR.^2,2);
    r_I_bounds = [min(r_I); max(r_I)];
    lambdas=diag(S);
    lambdas=lambdas(num_si+1);
else
    S_I=[];
    S_E = [S_hat']';
    SR  = dm-Dm_hat; 
    E_ch=SR;
    [U,S,V]=svd(E_ch);
    lambdas=diag(S);
    SSR = sum(SR.^2,2);  
    r_I=[];
    r_I_bounds=[];
end

%Conf. Intervals for S vectors.
 A=inv([ck,r_I]'*[ck,r_I]);
 A_=diag(A);
 e_T_e=diag(SR'*SR)';
 % 1.96 = 95 %  CI in a t-distribution
 S_E_CONF_INT=1.96*sqrt(repmat(e_T_e,size(A_,1),1).*repmat(A_,1,size(e_T_e,2)));

eiot_obj.S_hat          = S_hat;
eiot_obj.S_I            = S_I';
eiot_obj.S_E            = S_E;
eiot_obj.E_ch           = E_ch;
eiot_obj.r_I            = r_I;
eiot_obj.r_I_bounds     = r_I_bounds;
eiot_obj.SR             = SR;
eiot_obj.SSR            = SSR;
eiot_obj.num_si         = num_si;
eiot_obj.num_e_sI       = 0;
eiot_obj.abs_max_exc_ri = NaN;
eiot_obj.S_E_CONF_INT   = S_E_CONF_INT;
eiot_obj.lambdas        = lambdas;


