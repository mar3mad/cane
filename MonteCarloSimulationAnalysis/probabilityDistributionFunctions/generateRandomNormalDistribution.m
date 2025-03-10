function distribution = generateRandomNormalDistribution ...
    (meanValue, standardDeviation, numSamples)
%% Licensing
%
% License:         BSD License
%                  cane Multiphysics default license: cane/license.txt
%
% Main authors:    Andreas Apostolatos
%
%% Function documentation
%
% Returns an array of random numbers generated based on the normal 
% (Gaussian) distribution.
%
%             Input :
%         meanValue : Mean value of the distribution
% standardDeviation : Standard deviation of the distribution
%        numSamples : Number of samples
%
%            Output :
%      distribution : Random vector of noSamples normally distributed
%       
%% Function main body
distribution = normrnd(meanValue, standardDeviation, numSamples, 1);

end
