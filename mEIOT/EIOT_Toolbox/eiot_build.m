function [eiot_obj]=eiot_build(dm,ck,varargin)
% [eiot_obj]=eiot_build(dm,ck,<num_si_u,rk>)
%Inputs
% dm       : Matrix - pre-processed  mixture spectra [o x l]
% ck       : Maxtix - known concentrations           [o x n]
%
% Optional Inputs
% num_si_u : Scalar - Number of UNsupervised non-chemical interferences to identify 
%            [num_si_u=0 if argument is not sent] 
%
% rk       : Matrix - Values of known non-chemical interferences   [m x l]

%Outputs:
% eiot_obj.S_hat      : Estimation of apparent pure component spectra
% eiot_obj.S_I        : Non-chemical signatures [Supervised Unsupervised Exclusives]
% eiot_obj.S_E        : Extended set of signatures (chemical and non)
% eiot_obj.E_ch       : Spectral error after chemical deflation
% eiot_obj.r_I        : Strength of non-chemical interferences [Supervised Unsupervised Exclusive]
% eiot.rk_stats       : Mean and standard deviation of rk
% eiot_obj.SR         : Spectral residual
% eiot.lambdas        : eigenvalues of residual after chemical information and supervised interferences
% eiot_obj.SSR        : Squared sum of spectral residuals;
% eiot_obj.M          : Mahalanobis distance of r_I

    if     nargin==3
        num_si_u = varargin{1};
        rk=[];
    elseif nargin==4
        num_si_u = varargin{1};
        rk       = varargin{2};
    else
        num_si_u=0;
        rk=[];
    end

    if ~isempty(rk)
     [eiot_obj] = eiot_build_supervised(dm,ck,rk,num_si_u);
    else
     [eiot_obj] = eiot_build_unsup(dm,ck,num_si_u);
    end
end


function  [eiot_obj] = eiot_build_unsup(dm,ck,num_si)
    %function [eiot_obj] = eiotv2_build_unsup(Dm,Ck,num_si)
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
    eiot_obj.SR             = SR;
    eiot_obj.SSR            = SSR;
    eiot_obj.num_si         = num_si;
    eiot_obj.num_e_si       = 0;
    eiot_obj.S_E_CONF_INT   = S_E_CONF_INT;
    eiot_obj.lambdas        = lambdas;
end

function  [eiot_obj] = eiot_build_supervised(dm,ck,rk,num_si_u)
%function [eiot_obj] = eiot_build_supervised(dm,ck,rk,num_si_u)
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
num_e_si=0;  %Forcing exclusive signatures to zero, as they are non-relevant for now

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
eiot_obj.num_e_si   = num_e_si;
else
eiot_obj.num_e_si   = 0;
end

end
