function [r_hat,ri_hat,ssr] = eiot_calc(dm,eiot_obj,varargin)
%        [r_hat,ri_hat,ssr] = eiot_calc(dm,eiot_obj,<sum_r_nrs,rk>)
%Inputs:
% dm        : Vector of measure spectra (lambda x 1)
%
% eiot_obj  : Structure given by either of:
%             eiot_build.m
%             eiot_build_supervised.m
%
% Optional BUT IMPORTANT:
% sum_r_nrs : Summation of mass fraction for non-resolved species
%
% It is assumed that the order of signatures in S_E is:
% [Apparent_Pure_Spectra  Non_Chemical_Int  Ex_Non_Chem_Int]
%
%Outputs:
% c_E_hat: Concentrations/Strengths in the same order as in S_E
%
% ssr : Squared Spectral Residual, sum of squares of residuals after EIOT
%       deflation.


%default is no non-resolved species
sum_r_nrs=0;
rk=NaN;
rk_given=0;


if nargin ==3
    sum_r_nrs = varargin{1};
end
if nargin ==4
    sum_r_nrs = varargin{1};
    rk        = varargin{2};
    rk_given=1;
end

if isscalar(rk)
    if isnan(rk)
        rk_given=0;
    end
end

if size(dm,1)==1
    dm=dm';   % Assure dm has the correct orientation [ column vector ]
end

if rk_given
    rk = rk -  eiot_obj.rk_stats.mean;
    rk = rk ./ eiot_obj.rk_stats.std;
end
    
if eiot_obj.num_e_sI == 0 

    H   =  eiot_obj.S_E*eiot_obj.S_E';
    f   = -(dm'*eiot_obj.S_E');
    A   = [];
    b   = [];
    Aeq = [ones(1,size(eiot_obj.S_E,1)-eiot_obj.num_si) zeros(1,eiot_obj.num_si)];
%    Aeq = [ones(1,size(eiot_obj.S_hat,1)) zeros(1,eiot_obj.num_sI)];                     
    beq = 1-sum_r_nrs;
    if rk_given
        Aeq_ = diag(eiot_obj.index_rk_eq);
        Aeq_ = Aeq_(sum(Aeq_,2)==1,:);
        beq_ = rk';
        Aeq = [Aeq; Aeq_];
        beq = [beq; beq_];
    end
    c_E_guess =ones(1,size(eiot_obj.S_hat,1))/size(eiot_obj.S_hat,1);
    c_E_guess = [c_E_guess,ones(1,size(eiot_obj.S_I,1))];
    c_E_hat = quadprog(H,f,A,b,Aeq,beq);
    r_hat   = c_E_hat(1:size(eiot_obj.S_hat,1));
    ri_hat  = c_E_hat(size(eiot_obj.S_hat,1)+1:end);
    dm_hat  = eiot_obj.S_E'*c_E_hat;
    ssr     = sum((dm - dm_hat).^2);
    
else 
     me     = eiot_obj.num_e_sI;
     S_I_E =  eiot_obj.S_E(end-me+1:end,:);
     SE_aux = eiot_obj.S_E(1:end-me,:);
     
     SSR_m = [];
     for m = 1:me
            % Evaluate one Exclusive at a time
            SE_m    = [SE_aux;S_I_E(m,:)];
            H       =  SE_m*SE_m';
            f       = -(dm'*SE_m');
            A       =  [];
            b       =  [];
            Aeq     = [ones(1,size(SE_m,1)-eiot_obj.num_sI+me-1) zeros(1,eiot_obj.num_sI-me+1)];
            beq     = 1-sum_r_nrs;
            if rk_given
                Aeq_ = diag(eiot_obj.index_rk_eq);
                Aeq_ = Aeq_(sum(Aeq_,2)==1,:);
                beq_ = rk';
                Aeq = [Aeq; Aeq_];
                beq = [beq; beq_];
            end
            c_E_hat = quadprog(H,f,A,b,Aeq,beq);
            dm_hat  = SE_m'*c_E_hat;
            ssr_    = sum((dm - dm_hat).^2);
            SSR_m   = [SSR_m ; ssr_];
     end 
           [~,indx] = min(SSR_m);
              % Evaluate one Exclusive at a time
            SE_m    = [SE_aux;S_I_E(indx,:)];
            H       =  SE_m*SE_m';
            f       = -(dm'*SE_m');
            A       =  zeros(1,size(SE_m,1));
            A(end)  =  1;
            b       = eiot_obj.abs_max_exc_ri(indx);
            Aeq     = [ones(1,size(SE_m,1)-eiot_obj.num_sI+me-1) zeros(1,eiot_obj.num_sI-me+1)];
            beq     = 1-sum_r_nrs;
            c_E_hat = quadprog(H,f,A,b,Aeq,beq);
            dm_hat  = SE_m'*c_E_hat;
            ssr     = sum((dm - dm_hat).^2);
            rie     = zeros(me,1);
            rie(indx) = c_E_hat(end);
            c_E_hat = [c_E_hat(1:end-m);rie];
end % close end to main IF loop





