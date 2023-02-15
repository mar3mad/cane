%% Preamble
clc, clear;

%% Includes
addpath("../auxiliary/");

%% Define the problem parameters

paramStr.q0 = @(x,y) 0; % distributed load
% paramStr.E = 1e7; % Young's modulus
% paramStr.b = .1; % widthdlarrayf
% paramStr.h = .1; % height
% paramStr.A = paramStr.b*paramStr.h;
% paramStr.I = paramStr.b*(paramStr.h^3)/12; % moment of inertia
% paramStr.rho = 7850;

paramStr.Lambda = 1;

% Amplitude at the beam's ends
w0 = 0;

% Computational domain
propGeom.X0 = 0;
propGeom.XL = 1;
XComp = [propGeom.X0 propGeom.XL];

propGeom.Y0=0;
propGeom.YL=0.25;
YComp = [propGeom.Y0 propGeom.YL];

%% Define analytical solution for the provided data
syms X Y
w_analytical(X,Y) = sin(pi * X)*cos(2*pi*pi*Y);

%% Generate the training data

% points over which to enforce the boundary conditions by means of the loss
% of the neural network
numBC = 32;

% boundary conditions
% W(x=0 , y) = 0
XBC1 = propGeom.X0 * ones(1,numBC);
YBC1 = linspace(propGeom.Y0, propGeom.YL , numBC);
WBC1 = ones(1, numBC)*w0;

% W(x=1 , y) = 0
% XBC2 = propGeom.XL * ones(1,numBC);
% YBC2 = linspace(propGeom.Y0, propGeom.YL , numBC);
% WBC2 = ones(1, numBC)*w0;

XBC=XBC1;
YBC=YBC1;
WBC=WBC1;


% XBC = [XBC1 XBC2];
% YBC = [YBC1 YBC2];
% WBC = [WBC1 WBC2];

%% Create some random points between the computational interval
numInternColl = 32;
% % pointSet = sobolset(2);
% % points = net(pointSet,numInternColl);
% X = XComp(1, 1) + points(:,1)'*(XComp(1, 2) - XComp(1, 1));
% % X = linspace(0.1, 1.1, 10);
% T = Tstart + points(:,2)'*(Tend - Tstart);

X = linspace (propGeom.X0, propGeom.XL, numInternColl);
Y = linspace (propGeom.Y0, propGeom.YL, numInternColl);

[X,Y] = meshgrid(X,Y);

X = reshape(X ,[1 , numInternColl*numInternColl]);
Y = reshape(Y ,[1 , numInternColl*numInternColl]);

%% Deep learning model

% Number of layers
numLayers = 5;

% Number of neurons per layer
numNeurons = 128;

% initialize the weights and biases for the first fully connected operation
% The input layer has 1 neuron and numNeurons connections to the next layer
parameters.fc1_Weights = dlarray(sqrt(2/numNeurons)*randn([numNeurons 2]));
parameters.fc1_Bias = dlarray(zeros([numNeurons 1]));

% Initialize the weights and biases for the hidden fully connected 
% operations. Note that each hidden layer has numNeurons x numNeurons 
% connections
for layerNumber = 2:numLayers - 1
    name = "fc" + layerNumber;

    parameters.(name + "_Weights") = dlarray(sqrt(2/numNeurons/(2^(layerNumber-2)))*randn([numNeurons/(2^(layerNumber-1)) numNeurons/(2^(layerNumber-2))]));
    parameters.(name + "_Bias") = dlarray(zeros([numNeurons/(2^(layerNumber-1)) 1]));
end

% Initialize the final fully connected operation. The output layer has only 
% one neuron and returns the predicted solution w(x)

parameters.("fc" + numLayers + "_Weights") = dlarray(sqrt(2/numNeurons)*randn([1 numNeurons/(2^(numLayers-2))]));
parameters.("fc" + numLayers + "_Bias") = dlarray(zeros([1 1]));

% initialize the weights and biases for the first fully connected operation
% The input layer has 1 neuron and numNeurons connections to the next layer


%% Specify the optimization options
opts = optimoptions('fmincon', ...
    'HessianApproximation', {'lbfgs',50}, ...
    'MaxIterations', 10000 , ... % 7500
    'MaxFunctionEvaluations', 10000, ... % 7500
    'OptimalityTolerance', 1e-12, ...
    'SpecifyObjectiveGradient', true, ...
    'UseParallel', true, ...
    'Display', 'iter'); % iter

%% Train the network using fmincon

% Early Warmup
% learning parameters
propLearning.rhoR = 0.1;
propLearning.numR = numInternColl*numInternColl;
propLearning.rhoB = 1;
propLearning.numB = numBC * 2;
propLearning.rhoI = 1;
propLearning.numI = numBC ;


% extract data and names from the parameters
[parametersV,parameterNames,parameterSizes] = parameterStructToVector( ...
    parameters);
parametersV = extractdata(parametersV);

% Convert the training data to dlarray objects with format 'CB' (channel, batch).
dlX = dlarray(X,'CB');
dlY = dlarray(Y,'CB');
dlXBC = dlarray(XBC,'CB');
dlYBC = dlarray(YBC,'CB') ;
dlWBC = dlarray(WBC,'CB');

% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters, dlX, dlY, dlXBC, dlYBC, dlWBC, ...
    parameterNames, parameterSizes, paramStr, propLearning);

% Update the learnable parameters using the fmincon function
parametersV = fmincon(objFun, parametersV, [], [], [], [], [], [], [], ...
    opts);

% For prediction convert the vector of parameters to a structure
parameters = parameterVectorToStruct(parametersV, ...
    parameterNames, parameterSizes);

% Export network into dat file for future use
save ('2D_Beam.mat', 'parameters');

%% Auxiliary functions

%% Define the objective function that optimizes for the weights and biases
function [loss, gradientsV] = objectiveFunction ...
    (parametersV, dlX, dlY, dlXBC, dlYBC, dlWBC, parameterNames, parameterSizes, propStr, propLearning)

    % Convert parameters to structure of dlarray objects
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV, parameterNames, ...
        parameterSizes);
    
    % Evaluate model gradients and loss
    [gradients, loss] = dlfeval(@modelGradients, parameters, dlX, dlY, dlXBC, dlYBC, dlWBC, ...
        propStr, propLearning);    
    % Return loss and gradients for fmincon
    gradientsV = parameterStructToVector(gradients);
    gradientsV = extractdata(gradientsV);
    loss = extractdata(loss);
end

%% Function that returns the loss and its gradient with respect to the weights and biases
function [gradients,loss] = modelGradients(parameters, dlX, dlY, dlXBC, dlYBC, dlWBC, paramStr, propLearning)

    % Make predictions with the initial conditions
    W = model(parameters, dlX, dlY);

    
    % Plot the solution against the expected one from isogeometric analysis
    XHat = extractdata(dlX);
    YHat = extractdata(dlY);
    WHat = extractdata(W);
    XHat = reshape(XHat, [propLearning.numI,propLearning.numI]);
    YHat = reshape(YHat, [propLearning.numI,propLearning.numI]);
    WHat = reshape(WHat, [propLearning.numI,propLearning.numI]);
    
    surf(XHat(1,:),YHat(:,1), WHat);
    hold on
    fsurf(@(X,Y) sin(X.*pi  ).*cos(Y.*pi.*pi), [0 1 0 1],'--r', 'Linewidth',2);
    hold off
    xlabel('X');
    ylabel('Y');
    zlabel('X_0 + w');
    title("Beam's deformation");
    legend('PINN', 'Analytical');
    drawnow

    % Calculate the derivatives with respect to X
    Wx = dlgradient(sum(W,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxx = dlgradient(sum(Wx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxx = dlgradient(sum(Wxx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxxx = dlgradient(sum(Wxxx,'all'), dlX, 'EnableHigherDerivatives', true);

    Wy = dlgradient(sum(W,'all'), dlY, 'EnableHigherDerivatives', true);
    Wyy= dlgradient(sum(Wy,'all'), dlY, 'EnableHigherDerivatives', true);
    
    % Calculate lossF. Enforce the Euler-Bernoulli equation
    %     f = paramStr.E*paramStr.I*Wxxxx - + paramStr.rho*paramStr.A*Wtt - paramStr.q0(dlX,dlT);
    f = paramStr.Lambda.*Wxxxx + Wyy - paramStr.q0(dlX,dlY);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = huber(f, zeroTarget);
    
    % Calculate lossU0. Enforce boundary conditions.
    dlU0Pred = model(parameters, dlXBC , dlYBC);
    lossU0 = huber(dlU0Pred, dlWBC);

    % Calculate the second derivatives of the boundary conditions with 
    % respect to Y
    dlWBCy = dlgradient(sum(dlU0Pred ,'all'), dlYBC, 'EnableHigherDerivatives', true);
    dlWBCyy = dlgradient(sum(dlWBCy,'all'), dlYBC, 'EnableHigherDerivatives', true);

    % Calculate the second derivatives of the boundary conditions with 
    % respect to X
    dlWBCx = dlgradient(sum(dlU0Pred ,'all'), dlXBC, 'EnableHigherDerivatives', true);
    dlWBCxx = dlgradient(sum(dlWBCx,'all'), dlXBC, 'EnableHigherDerivatives', true);

    % Enforce the second derivative of the displacement field to zero (X)
    zeroTargetDDwDDX = zeros(size(dlWBCxx), 'like', f);
    lossDDwDDX = huber(dlWBCxx, zeroTargetDDwDDX);

     % Enforce the second derivative of the displacement field to zero (Y)
    zeroTargetDDwDDY = zeros(size(dlWBCyy), 'like', f);
    lossDDwDDY = huber(dlWBCyy, zeroTargetDDwDDY);

     % Enforce the first derivative of the displacement field to zero (Y)
    zeroTargetDwDY = zeros(size(dlWBCy), 'like', f);
    lossDwDY = huber(dlWBCy, zeroTargetDwDY);

     % Enforce the first derivative of the displacement field to zero (Y)
    zeroTargetDwDX = zeros(size(dlWBCx), 'like', f);
    lossDwDX = huber(dlWBCx, zeroTargetDwDX);
    
    % Combine losses into the general loss function
    loss = propLearning.rhoR*(1/propLearning.numR)*lossF + ...
        propLearning.rhoB*(1/propLearning.numB)*(lossU0 + lossDDwDDX+lossDDwDDY)+ ...
        propLearning.rhoI*(1/propLearning.numI)*(lossUI + lossDwDY+lossDwDX);
    
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss, parameters);

    
end



%% Feed-forward (model) function for the neural network
function dlU = model(parameters, dlXBC, dlYBC)

    dlXT = [dlXBC;dlYBC];
    numLayers = numel(fieldnames(parameters))/2;
    
    % First fully connect operation.
    weights = parameters.fc1_Weights;
    bias = parameters.fc1_Bias;
    dlU = fullyconnect(dlXT,weights,bias);
    
    % tanh and fully connect operations for remaining layers.
    for i = 2:numLayers
        name = "fc" + i;
    
        dlU = tanh(dlU);
    
        weights = parameters.(name + "_Weights");
        bias = parameters.(name + "_Bias");
        dlU = fullyconnect(dlU, weights, bias);
    end
    
end