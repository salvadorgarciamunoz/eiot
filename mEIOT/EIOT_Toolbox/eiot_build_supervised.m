function  [eiot_obj,varargout] = eiot_build_supervised(dm,ck,rk,num_si_u)
%function [eiot_obj] = eiot_build_supervised(Dm,Ck,Rk,num_e_sI,num_sI_U)
%Inputs:
% dm       : Pre-processed matrix of mixture spectra
% ck       : Known concentrations of mixture spectra
% rk       : Known non-chemical interferences
%            one column of S_I is identified per column of Rk
% num_si_u : Number of UNsupervised non-chemical interferences to identify additional to those
%            identified from rk
%Outputs:
% eiot_obj.S_hat      : Estimation of apparent pure component spectra
% eiot_obj.S_I        : Non-chemical signatures [Supervised Unsupervised Exclusives]
% eiot_obj.S_E        : Extended set of signatures (chemical and non)
% eiot_obj.E_ch       : Spectral error after chemical deflation
% eiot_obj.r_I        : Strength of non-chemical interferences [Supervised Unsupervised Exclusive]
% eiot.rk_stats       : Mean and standard deviation of rk
% eiot_obj.SR         : Spectral residual
% eiot_obj.SSR        : Squared sum of spectral residuals;
% eiot.lambdas        : eigenvalues of residual after chemical information and supervised interferences

lambdas=[];
num_e_si=0;  %Forcing exclusive signatures to zero non-relevant for now

rk_stats.mean = mean(rk);
rk_stats.std  = std(rk);
  
rk_ = rk - repmat(mean(rk),size(rk,1),1);
rk_ = rk_ ./ repmat(std(rk_),size(rk,1),1);


if num_e_si >0
    rk=[rk_(:,1:size(rk,2)-num_e_si),rk(:,end-num_e_si+1:end)];
    rk_stats.mean(end-num_e_si+1:end)=0;
    rk_stats.std(end-num_e_si+1:end)=1;
else
   rk=rk_; 
end

Ck_aug  = [ck,rk];
num_sI  = size(rk,2)+num_si_u;
S_hat   = pinv(Ck_aug'*Ck_aug)*Ck_aug'*dm;
S_E_tmp = [S_hat];
S_I_S   = S_hat(size(ck,2)+1:end,:);  % Supervised non-chemical interferences
S_hat   = S_hat(1:size(ck,2),:);      % Apparent pure spectrum for chemical species

flag_force_no_exclusives=0;

if num_e_si==0
    S_I_E=[]; 
    r_I_E=[];
    r_I_S=rk;
elseif size(S_I_S,1)==num_e_si
    S_I_E=S_I_S;
    S_I_S=[];
    r_I_S=[];
    r_I_E=rk;
elseif size(S_I_S,1)>num_e_si
    S_I_E=S_I_S(end-num_e_si+1:end,:);
    S_I_S=S_I_S(1:end-num_e_si,:);
    r_I_E=rk(:,end-num_e_si+1:end);
    r_I_S=rk(:,1:end-num_e_si);
else
    S_I_E=[];
    r_I_E=[];
    flag_force_no_exclusives =1;  %if the given number of exclusive non-chem int is > size(Rk,2)
end


E_ch_tmp = dm- Ck_aug*S_E_tmp;

if num_si_u>0
    [U,S,V]=svd(E_ch_tmp);
    S_I_U   = V(:,1:num_si_u);
    S_I_U   = S_I_U';
    S_short = S(1:num_si_u,1:num_si_u);
    r_I_U   = U(:,1:num_si_u)*S_short;
    
    S_E = [S_hat;S_I_S;S_I_U;S_I_E];
    S_I = [S_I_S;S_I_U;S_I_E];
    r_I = [r_I_S,r_I_U,r_I_E];
    lambdas=diag(S);
    lambdas=lambdas(num_sI+1);
    
    index_rk_eq=[zeros(1,size(S_hat,1)),ones(1,size(S_I_S,1)),zeros(1,size(S_I_U,1)),ones(1,size(S_I_E,1))];
    
else
    S_E = S_E_tmp;
    S_I = [S_I_S;S_I_E];
    r_I = [r_I_S,r_I_E];
    [U,S,V]=svd(E_ch_tmp);
    lambdas=diag(S);
    index_rk_eq=[zeros(1,size(S_hat,1)),ones(1,size(S_I,1))];
end



SR             = dm-[ck,r_I]*S_E;
SSR            = sum(SR.^2,2);
abs_max_exc_ri = 1.5*range(r_I_E)/2;
E_ch  = dm- ck*S_hat;

%Confidence Intervals for S vectors
 A=pinv([ck,r_I]'*[ck,r_I]);
 A_=diag(A);
 e_T_e=diag(SR'*SR)';
 % 1.96 = 95 %  CI in a t-distribution
 S_E_CONF_INT=1.96*sqrt(repmat(e_T_e,size(A_,1),1).*repmat(A_,1,size(e_T_e,2)));
 
eiot_obj.S_hat        = S_hat;
eiot_obj.S_I          = S_I;
eiot_obj.S_E          = S_E;
eiot_obj.E_ch         = E_ch;
eiot_obj.r_I          = r_I;
eiot_obj.SR           = SR;
eiot_obj.SSR          = SSR;
eiot_obj.num_si       = num_sI;
eiot_obj.Rk_calc      = rk;
eiot_obj.S_E_CONF_INT = S_E_CONF_INT;
eiot_obj.rk_stats     = rk_stats;
eiot_obj.index_rk_eq  = index_rk_eq;
eiot_obj.lambdas      =lambdas;

if ~flag_force_no_exclusives
eiot_obj.num_e_sI   = num_e_si;
else
eiot_obj.num_e_sI   = 0;
end
eiot_obj.abs_max_exc_ri = abs_max_exc_ri;
