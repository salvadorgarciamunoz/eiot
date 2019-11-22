function [data] = eiot_snv(data)
if isvector(data)
    if size(data,1)~=1
        data=data';
    end
end
data=data';
data = data - repmat(mean(data),size(data,1),1);
data = data ./ repmat(std(data),size(data,1),1);
data = data';

