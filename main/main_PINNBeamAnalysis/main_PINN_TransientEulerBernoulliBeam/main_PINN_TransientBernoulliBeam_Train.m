%% Preamble
clc, clear;

%% Includes
addpath("../auxiliary/");

%% Define the problem parameters

paramStr.q0 = @(x,t) 0; % distributed load
% paramStr.E = 1e7; % Young's modulus
% paramStr.b = .1; % width
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

Tstart = 0;
Tend = 1;



%% Define analytical solution for the provided data
syms X T
w_analytical(X,T) = sin(pi * X)*cos(2*pi*pi*T);

%% Generate the training data

% points over which to enforce the boundary conditions by means of the loss
% of the neural network
numBC = 32;

% boundary conditions
% W(x=0 , t) = 0
XBC1 = propGeom.X0 * ones(1,numBC);
TBC1 = linspace(Tstart, Tend , numBC);
WBC1 = ones(1, numBC)*w0;

% W(x=1 , t) = 0
XBC2 = propGeom.XL * ones(1,numBC);
TBC2 = linspace(Tstart, Tend , numBC);
WBC2 = ones(1, numBC)*w0;

% W(x , t=0) = 0
XIC = linspace(propGeom.X0, propGeom.XL, numBC);
TIC = zeros(1, numBC);
WIC = sin(pi * linspace(propGeom.X0, propGeom.XL, numBC));

XBC = [XBC1 XBC2];
TBC = [TBC1 TBC2];
WBC = [WBC1 WBC2];

numBC2 = 256;

% boundary conditions
% W(x=0 , t) = 0
XBC21 = propGeom.X0 * ones(1,numBC2);
TBC21 = linspace(Tstart, Tend , numBC2);
WBC21 = ones(1, numBC2)*w0;

% W(x=1 , t) = 0
XBC22 = propGeom.XL * ones(1,numBC2);
TBC22 = linspace(Tstart, Tend , numBC2);
WBC22 = ones(1, numBC2)*w0;

% W(x , t=0) = 0
XIC2 = linspace(propGeom.X0, propGeom.XL, numBC2);
TIC2 = zeros(1, numBC2);
WIC2 = sin(pi * linspace(propGeom.X0, propGeom.XL, numBC2));

XBC2 = [XBC21 XBC22];
TBC2 = [TBC21 TBC22];
WBC2 = [WBC21 WBC22];



%% Create some random points between the computational interval
numInternColl = 32;
% % pointSet = sobolset(2);
% % points = net(pointSet,numInternColl);
% X = XComp(1, 1) + points(:,1)'*(XComp(1, 2) - XComp(1, 1));
% % X = linspace(0.1, 1.1, 10);
% T = Tstart + points(:,2)'*(Tend - Tstart);

X = linspace (propGeom.X0, propGeom.XL, numInternColl);
T = linspace (Tstart, Tend, numInternColl);

[X,T] = meshgrid(X,T);

X = reshape(X ,[1 , numInternColl*numInternColl]);
T = reshape(T ,[1 , numInternColl*numInternColl]);

numInternColl2 = 256;
% % pointSet = sobolset(2);
% % points = net(pointSet,numInternColl);
% X = XComp(1, 1) + points(:,1)'*(XComp(1, 2) - XComp(1, 1));
% % X = linspace(0.1, 1.1, 10);
% T = Tstart + points(:,2)'*(Tend - Tstart);

X2 = linspace (propGeom.X0, propGeom.XL, numInternColl2);
T2 = linspace (Tstart, Tend, numInternColl2);

[X2,T2] = meshgrid(X2,T2);

X2 = reshape(X2 ,[1 , numInternColl2*numInternColl2]);
T2 = reshape(T2 ,[1 , numInternColl2*numInternColl2]);

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
dlT = dlarray(T,'CB');
dlXBC = dlarray(XBC,'CB');
dlTBC = dlarray(TBC,'CB') ;
dlWBC = dlarray(WBC,'CB');

dlXIC = dlarray(XIC,'CB');
dlTIC = dlarray(TIC,'CB') ;
dlWIC = dlarray(WIC,'CB');

% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters, dlX, dlT, dlXBC, dlTBC, dlWBC,dlXIC, dlTIC, dlWIC, ...
    parameterNames, parameterSizes, paramStr, propLearning);

% Update the learnable parameters using the fmincon function
parametersV = fmincon(objFun, parametersV, [], [], [], [], [], [], [], ...
    opts);

%Fine Tuning
dlX2 = dlarray(X2,'CB');
dlT2 = dlarray(T2,'CB');
dlXBC2 = dlarray(XBC2,'CB');
dlTBC2 = dlarray(TBC2,'CB') ;
dlWBC2 = dlarray(WBC2,'CB');

dlXIC2 = dlarray(XIC2,'CB');
dlTIC2 = dlarray(TIC2,'CB') ;
dlWIC2 = dlarray(WIC2,'CB');

% learning parameters
propLearning.rhoR = 100;
propLearning.numR = numInternColl2*numInternColl2;
propLearning.rhoB = 1000;
propLearning.numB = numBC2 * 2;
propLearning.rhoI = 1000;
propLearning.numI = numBC2 ;

% Create a function handle with one input that defines the objective function.
objFun = @(parameters) objectiveFunction(parameters, dlX2, dlT2, dlXBC2, dlTBC2, dlWBC2, dlXIC2, dlTIC2, dlWIC2, ...
    parameterNames, parameterSizes, paramStr, propLearning);

% Update the learnable parameters using the fmincon function
parametersV = fmincon(objFun, parametersV, [], [], [], [], [], [], [], ...
    opts);

% For prediction convert the vector of parameters to a structure
parameters = parameterVectorToStruct(parametersV, ...
    parameterNames, parameterSizes);

% Export network into dat file for future use
save ('1D_Beam.mat', 'parameters');

% layers=[fullyConnectedLayer(1,'Name','inputLayer')
%     tanh()];
% net = dlnetwork(layers);
% exportONNXNetwork(net);

%% Evaluate the model's accuracy

% % create a set of collocation points in the domain to predict the solution
% XTest = linspace(XComp(1, 1), XComp(1, 2), 100);
% dlXTest = dlarray(XTest, 'CB');
% 
% % predict the solution using the trained model
% dlWTest = model(parameters, dlXTest);
% WTest = extractdata(dlWTest);
% 
% figure(1)
% plot(XTest, WTest)
% axis equal




%% Auxiliary functions

%% Define the objective function that optimizes for the weights and biases
function [loss, gradientsV] = objectiveFunction ...
    (parametersV, dlX, dlT, dlXBC, dlTBC, dlWBC,dlXIC, dlTIC, dlWIC, parameterNames, parameterSizes, propStr, propLearning)

    % Convert parameters to structure of dlarray objects
    parametersV = dlarray(parametersV);
    parameters = parameterVectorToStruct(parametersV, parameterNames, ...
        parameterSizes);
    
    % Evaluate model gradients and loss
    [gradients, loss] = dlfeval(@modelGradients, parameters, dlX, dlT, dlXBC, dlTBC, dlWBC,dlXIC, dlTIC, dlWIC, ...
        propStr, propLearning);    
    % Return loss and gradients for fmincon
    gradientsV = parameterStructToVector(gradients);
    gradientsV = extractdata(gradientsV);
    loss = extractdata(loss);
end

%% Function that returns the loss and its gradient with respect to the weights and biases
function [gradients,loss] = modelGradients(parameters, dlX, dlT, dlXBC, dlTBC, dlWBC,dlXIC, dlTIC, dlWIC, paramStr, propLearning)

    % Make predictions with the initial conditions
    W = model(parameters, dlX, dlT);

    
    % Plot the solution against the expected one from isogeometric analysis
    XHat = extractdata(dlX);
    THat = extractdata(dlT);
    WHat = extractdata(W);
    XHat = reshape(XHat, [propLearning.numI,propLearning.numI]);
    THat = reshape(THat, [propLearning.numI,propLearning.numI]);
    WHat = reshape(WHat, [propLearning.numI,propLearning.numI]);
    
%     surf(XHat(1,:),THat(:,1), WHat);
%     hold on
%     fsurf(@(X,T) sin(X.*pi  ).*cos(T.*pi.*pi), [0 1 0 1],'--r', 'Linewidth',2);
%     hold off
%     xlabel('X');
%     ylabel('T');
%     zlabel('X_0 + w');
%     title("Beam's deformation");
%     legend('PINN', 'Analytical');
%     drawnow

    % Calculate the derivatives with respect to X
    Wx = dlgradient(sum(W,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxx = dlgradient(sum(Wx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxx = dlgradient(sum(Wxx,'all'), dlX, 'EnableHigherDerivatives', true);
    Wxxxx = dlgradient(sum(Wxxx,'all'), dlX, 'EnableHigherDerivatives', true);

    Wt = dlgradient(sum(W,'all'), dlT, 'EnableHigherDerivatives', true);
    Wtt= dlgradient(sum(Wt,'all'), dlT, 'EnableHigherDerivatives', true);
    
    % Calculate lossF. Enforce the Euler-Bernoulli equation
    %     f = paramStr.E*paramStr.I*Wxxxx - + paramStr.rho*paramStr.A*Wtt - paramStr.q0(dlX,dlT);
    f = paramStr.Lambda.*Wxxxx + Wtt - paramStr.q0(dlX,dlT);
    zeroTarget = zeros(size(f), 'like', f);
    lossF = huber(f, zeroTarget);
    
    % Calculate lossU0. Enforce boundary conditions.
    dlU0Pred = model(parameters, dlXBC , dlTBC);
    lossU0 = huber(dlU0Pred, dlWBC);

     % Calculate lossUI. Enforce Initial conditions.
    dlUIPred = model(parameters, dlXIC , dlTIC);
    lossUI = huber(dlUIPred, dlWIC);

    % Calculate the second derivatives of the boundary conditions with 
    % respect to X
    dlU0X = model(parameters, dlXBC , dlTBC);
    dlWBCx = dlgradient(sum(dlU0X,'all'), dlXBC, 'EnableHigherDerivatives', true);
    dlWBCxx = dlgradient(sum(dlWBCx,'all'), dlXBC, 'EnableHigherDerivatives', true);

    % Enforce the second derivative of the displacement field to zero
    zeroTargetDDwDDX = zeros(size(dlWBCxx), 'like', f);
    lossDDwDDX = huber(dlWBCxx, zeroTargetDDwDDX);

    % Calculate the first derivateves of the initial conditions with
    % respect to T
    dlU0T = model(parameters, dlXIC , dlTIC);
    dlWBCt = dlgradient(sum(dlU0T,'all'), dlTIC, 'EnableHigherDerivatives', true);

     % Enforce the first derivative of the displacement field to zero
    zeroTargetDwDT = zeros(size(dlWBCt), 'like', f);
    lossDwDT = huber(dlWBCt, zeroTargetDwDT);
    
    % Combine losses into the general loss function
    loss = propLearning.rhoR*(1/propLearning.numR)*lossF + ...
        propLearning.rhoB*(1/propLearning.numB)*(lossU0 + lossDDwDDX)+ ...
        propLearning.rhoI*(1/propLearning.numI)*(lossUI + lossDwDT);
    
    % Calculate gradients with respect to the learnable parameters.
    gradients = dlgradient(loss, parameters);

    
end



%% Feed-forward (model) function for the neural network
function dlU = model(parameters, dlXBC, dlTBC)

    dlXT = [dlXBC;dlTBC];
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