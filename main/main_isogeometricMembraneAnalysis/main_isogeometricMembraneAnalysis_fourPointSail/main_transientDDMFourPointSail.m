%% Licensing
%
% License:         BSD License
%                  cane Multiphysics default license: cane/license.txt
%
% Main authors:    Andreas Apostolatos
%
%% Script documentation
% 
% Task : Transient simulation of a 3-patch four-point sail subject to 
%        surface distributed load
%
% Date : 10.11.2016
%
%% Preamble
clear;
clc;

%% Includes

% Add general math functions
addpath('../../../generalMath/');

% Add general auxiliary functions
addpath('../../../auxiliary/');

% Add system solvers
addpath('../../../equationSystemSolvers/');

% Add efficient computation functions
addpath('../../../efficientComputation/');

% Add transient analysis solvers
addpath('../../../transientAnalysis/');

% Add all functions related to the Computer-Aided Geometric Design (GACD) kernel
addpath('../../../CAGDKernel/CAGDKernel_basisFunctions',...
        '../../../CAGDKernel/CAGDKernel_geometryResolutionRefinement/',...
        '../../../CAGDKernel/CAGDKernel_baseVectors/',...
        '../../../CAGDKernel/CAGDKernel_graphics/',...
        '../../../CAGDKernel/CAGDKernel_BSplineCurve/',...
        '../../../CAGDKernel/CAGDKernel_BSplineSurface/');
    
% Add all functions related to the isogeometric Kirchhoff-Love shell formulation
addpath('../../../isogeometricThinStructureAnalysis/graphicsSinglePatch/',...
        '../../../isogeometricThinStructureAnalysis/graphicsMultipatches/',...
        '../../../isogeometricThinStructureAnalysis/loads/',...
        '../../../isogeometricThinStructureAnalysis/solutionMatricesAndVectors/',...
        '../../../isogeometricThinStructureAnalysis/solvers/',...
        '../../../isogeometricThinStructureAnalysis/metrics/',...
        '../../../isogeometricThinStructureAnalysis/auxiliary/',...
        '../../../isogeometricThinStructureAnalysis/postprocessing/',...
        '../../../isogeometricThinStructureAnalysis/BOperatorMatrices/',...
        '../../../isogeometricThinStructureAnalysis/penaltyDecompositionKLShell/',...
        '../../../isogeometricThinStructureAnalysis/penaltyDecompositionMembrane/',...
        '../../../isogeometricThinStructureAnalysis/lagrangeMultipliersDecompositionKLShell/',...
        '../../../isogeometricThinStructureAnalysis/nitscheDecompositionMembrane/',...
        '../../../isogeometricThinStructureAnalysis/errorComputation/',...
        '../../../isogeometricThinStructureAnalysis/output/',...
        '../../../isogeometricThinStructureAnalysis/transientAnalysis/',...
        '../../../isogeometricThinStructureAnalysis/initialConditions/',...
        '../../../isogeometricThinStructureAnalysis/weakDBCMembrane/',...
        '../../../isogeometricThinStructureAnalysis/formFindingAnalysis/');

%% Read the geometry from the results of a form finding analysis

% Read the data
fileName = 'FoFiDDMFourPointSail';
meshSize = 'Fine'; % Coarse, Fine
if ~strcmp(meshSize, 'Coarse') && ~strcmp(meshSize, 'Fine')
    error('meshSize can be either set to coarse or fine');
end
FoFiGeo = importdata(['./data_FoFiPool/' 'data_' fileName meshSize '.mat']);

% Patch 1 :
% _________

% Polynomial orders
p1 = FoFiGeo.BSplinePatches{1}.p;
q1 = FoFiGeo.BSplinePatches{1}.q;

% Knot vectors
Xi1 = FoFiGeo.BSplinePatches{1}.Xi;
Eta1 = FoFiGeo.BSplinePatches{1}.Eta;

% Control Point coordinates and weights
CP1 = FoFiGeo.BSplinePatches{1}.CP;

% Flag on whether the basis is a B-Spline or a NURBS
isNURBS1 = FoFiGeo.BSplinePatches{1}.isNURBS;

% Patch 2 :
% _________

% Polynomial orders
p2 = FoFiGeo.BSplinePatches{2}.p;
q2 = FoFiGeo.BSplinePatches{2}.q;

% Knot vectors
Xi2 = FoFiGeo.BSplinePatches{2}.Xi;
Eta2 = FoFiGeo.BSplinePatches{2}.Eta;

% Control Point coordinates and weights
CP2 = FoFiGeo.BSplinePatches{2}.CP;

% Flag on whether the basis is a B-Spline or a NURBS
isNURBS2 = FoFiGeo.BSplinePatches{2}.isNURBS;

% Patch 3 :
% _________

% Polynomial orders
p3 = FoFiGeo.BSplinePatches{3}.p;
q3 = FoFiGeo.BSplinePatches{3}.q;

% Knot vectors
Xi3 = FoFiGeo.BSplinePatches{3}.Xi;
Eta3 = FoFiGeo.BSplinePatches{3}.Eta;

% Control Point coordinates and weights
CP3 = FoFiGeo.BSplinePatches{3}.CP;

% Flag on whether the basis is a B-Spline or a NURBS
isNURBS3 = FoFiGeo.BSplinePatches{3}.isNURBS;

%% Material constants

% general parameters
EYoung = 8e+8;
nue = .4;
thickness = 1e-3;
sigma0 = 3e+3/thickness;
prestress.voigtVector = [sigma0
                         sigma0
                         0];
density = 800;

% Patch 1 :
% _________

% Young's modulus
parameters1.E = EYoung;

% Poisson ratio
parameters1.nue = nue;

% Thickness of the membrane
parameters1.t = thickness;

% Density of the membrane
parameters1.rho = density;

% Prestress of the membrane
parameters1.prestress = prestress;

% Patch 2 :
% _________

% Young's modulus
parameters2.E = EYoung;

% Poisson ratio
parameters2.nue = nue;

% Thickness of the plate
parameters2.t = thickness;

% Density of the membrane
parameters2.rho = density;

% Prestress of the membrane
parameters2.prestress = prestress;

% Patch 3 :
% _________

% Young's modulus
parameters3.E = EYoung;

% Poisson ratio
parameters3.nue = nue;

% Thickness of the plate
parameters3.t = thickness;

% Density of the membrane
parameters3.rho = density;

% Prestress of the membrane
parameters3.prestress = prestress;

% Cable :
% _______

parametersCable.E = 1.6e+11;
parametersCable.radiusCS = 12e-3/2;
parametersCable.areaCS = pi*parametersCable.radiusCS^2;
parametersCable.rho = 8300;
parametersCable.prestress = 6e+4/parametersCable.areaCS;

%% GUI

% Case name
caseName = 'transientDDM3PatchesFourPointSail';

% Analysis type
propAnalysis.type = 'isogeometricMembraneAnalysis';

% Define equation system solver
solve_LinearSystem = @solve_LinearSystemMatlabBackslashSolver;

% Co-simulation with Empire
propEmpireCoSimulation.isCoSimulation = false; % Flag on whether co-simulation with EMPIRE is assumed
propEmpireCoSimulation.isInterfaceLayer = false; % Flag on whether the matlab client is used as an interface layer
propEmpireCoSimulation.strMatlabXml = 'undefined'; % Name of the xml file for Matlab

% Choose a method for the application of weak Dirichlet boundary conditions and the multipatch coupling
method = 'Penalty';
if ~strcmp(method, 'Penalty') && ~strcmp(method, 'LagrangeMultipliers') && ...
        ~strcmp(method, 'Mortar') && ~strcmp(method, 'AugmentedLagrangeMultipliers') && ...
        ~strcmp(method, 'Nitsche')
    error('%s is not a valid method (Nitsche, Penalty, LagrangeMultipliers, AugmentedLagrangeMultipliers)', method);
end

% Integration scheme
% type = 'default' : default FGI integration element-wise
% type = 'user' : manual choice of the number of Gauss points

% Patch 1 :
% _________

int1.type = 'default';
if strcmp(int1.type, 'user')
    int1.xiNGP = 6;
    int1.etaNGP = 6;
    int1.xiNGPForLoad = 6;
    int1.etaNGPForLoad = 6;
    int1.nGPForLoad = 6;
    int1.nGPError = 12;
end

% Patch 2 :
% _________

int2.type = 'default';
if strcmp(int2.type, 'user')
    int2.xiNGP = 6;
    int2.etaNGP = 6;
    int2.xiNGPForLoad = 6;
    int2.etaNGPForLoad = 6;
    int2.nGPForLoad = 6;
    int2.nGPError = 12;
end

% Patch 3 :
% _________

int3.type = 'default';
if strcmp(int3.type, 'user')
    int3.xiNGP = 6;
    int3.etaNGP = 6;
    int3.xiNGPForLoad = 6;
    int3.etaNGPForLoad = 6;
    int3.nGPForLoad = 6;
    int3.nGPError = 12;
end

% Interface integration :
% _______________________

intC.type = 'user';
intC.method = 'Nitsche';
if strcmp(intC.type, 'user')
    if strcmp(intC.method, 'lagrangeMultipliers')
        intC.nGP1 = 16;
        intC.nGP2 = 16;
    else
        intC.noGPs = 16;
    end
    intC.nGPError = 16;
end

% On the graphics
graph.index = 1;

% On the postprocessing:
% .postprocConfig : 'reference','current','referenceCurrent'
graph.postprocConfig = 'current';

% Plot strain or stress field
% .resultant: 'displacement','strain','curvature','force','moment','shearForce'
graph.resultant = 'displacement';

% Component of the resultant to plot
% .component: 'x','y','z','2norm','1','2','12','1Principal','2Principal'
graph.component = '2norm';

% Define the coupling properties

% Patch 1 :
% _________

Dm1 = parameters1.E*parameters1.t/(1-parameters1.nue^2)*...
      [1              parameters1.nue 0
       parameters1.nue 1              0
       0               0              (1 - parameters1.nue)/2];
Db1 = parameters1.t^2/12*Dm1;
   
% Patch 2 :
% _________

Dm2 = parameters2.E*parameters2.t/(1 - parameters2.nue^2)*...
      [1              parameters2.nue 0
       parameters2.nue 1              0
       0               0              (1 - parameters2.nue)/2];
Db2 = parameters2.t^2/12*Dm2;

% Patch 3 :
% _________

Dm3 = parameters3.E*parameters3.t/(1 - parameters3.nue^2)*...
      [1              parameters3.nue 0
       parameters3.nue 1              0
       0               0              (1 - parameters3.nue)/2];
Db3 = parameters3.t^2/12*Dm3;

% Assign the penalty factors

% Function handle to writing out the results
% propOutput.writeOutput = @writeResults4Carat;
% propOutput.writeOutput = @writeResults4GiD;
propOutput.writeOutput = @writeResults4GiD;
propOutput.writeFrequency = 1;

% Postprocessing
propPostproc.resultant = {'displacement'};
propPostproc.computeResultant = {'computeDisplacement'};

% Co-simulation with Empire
isCosimulationWithEmpire = false;
strMatlabXml = 'undefined';

% Start and end time of the simulation
TStart = 0.0;
TEnd = 1.0;

%% Refinement

%%%%%%%%%%%%%%%%%%%%
% Degree elevation %
%%%%%%%%%%%%%%%%%%%%

% Patch 1 :
% _________

a = 0;
tp1 = a;
tq1 = a;
[Xi1, Eta1, CP1, p1, q1] = degreeElevateBSplineSurface ...
    (p1, q1, Xi1, Eta1, CP1, tp1, tq1, '');

% Patch 2 :
% _________

b = 0;
tp2 = b;
tq2 = b;
[Xi2, Eta2, CP2, p2, q2] = degreeElevateBSplineSurface ...
    (p2, q2, Xi2, Eta2, CP2, tp2, tq2, '');

% Patch 3 :
% _________

c = 0;
tp3 = c;
tq3 = c;
[Xi3, Eta3, CP3, p3, q3] = degreeElevateBSplineSurface ...
    (p3, q3, Xi3, Eta3, CP3, tp3, tq3, '');

%%%%%%%%%%%%%%%%%%%%
% Knot insertion   %
%%%%%%%%%%%%%%%%%%%%

% Patch 1 :
% _________

noKnotsXi1 = 0;
noKnotsEta1 = noKnotsXi1;
[Xi1, Eta1, CP1] = knotRefineUniformlyBSplineSurface ...
    (p1, Xi1, q1, Eta1, CP1, noKnotsXi1, noKnotsEta1, '');

% Patch 2 :
% _________

noKnotsXi2 = 0;
noKnotsEta2 = noKnotsXi2;
[Xi2, Eta2, CP2] = knotRefineUniformlyBSplineSurface ...
    (p2, Xi2, q2, Eta2, CP2, noKnotsXi2, noKnotsEta2, '');

% Patch 3 :
% _________

noKnotsXi3 = 0;
noKnotsEta3 = noKnotsXi3;
[Xi3, Eta3, CP3] = knotRefineUniformlyBSplineSurface ...
    (p3, Xi3, q3, Eta3, CP3, noKnotsXi3, noKnotsEta3, '');

%% Boundary conditions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dirichlet boundary conditions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Patch 1 :
% _________

% Homogeneous Dirichlet boundary conditions
homDOFs1 = [];
xisup1 = [0 0];   etasup1 = [0 1];
for dir = 1:3
    homDOFs1 = findDofs3D ...
        (homDOFs1, xisup1, etasup1, dir, CP1);
end
xisup1 = [1 1];   etasup1 = [0 0];
for dir = 1:3
    homDOFs1 = findDofs3D ...
        (homDOFs1, xisup1, etasup1, dir, CP1);
end
xisup1 = [1 1];   etasup1 = [1 1];
for dir = 1:3
    homDOFs1 = findDofs3D ...
        (homDOFs1, xisup1, etasup1, dir, CP1);
end

% Inhomogeneous Dirichlet boundary conditions
inhomDOFs1 = [];
valuesInhomDOFs1 = [];

% Weak homogeneous Dirichlet boundary conditions
weakDBC1.noCnd = 0;

% Embedded cables
cables1.No = 2;
cables1.xiExtension = {[0 1] [0 1]};
cables1.etaExtension = {[0 0] [1 1]};
cables1.parameters = {parametersCable parametersCable parametersCable parametersCable};
cables1.int.type = 'default';
% cables1.int.type = 'user';
cables1.int.noGPs = 16;

% Patch 2 :
% _________

% Homogeneous Dirichlet boundary conditions
homDOFs2 = [];
xisup2 = [0 1];   etasup2 = [0 0];
for dir = 1:3
    homDOFs2 = findDofs3D ...
        (homDOFs2, xisup2, etasup2, dir, CP2);
end
xisup2 = [0 1];   etasup2 = [1 1];
for dir = 1:3
    homDOFs2 = findDofs3D ...
        (homDOFs2, xisup2, etasup2, dir, CP2);
end

% Inhomogeneous Dirichlet boundary conditions
inhomDOFs2 = [];
valuesInhomDOFs2 = [];

% Weak homogeneous Dirichlet boundary conditions
weakDBC2.noCnd = 0;

% Embedded cables
cables2.No = 0;

% Patch 3 :
% _________

% Homogeneous Dirichlet boundary conditions
homDOFs3 = [];
xisup3 = [0 0];   etasup3 = [0 0];
for dir = 1:3
    homDOFs3 = findDofs3D ...
        (homDOFs3, xisup3, etasup3, dir, CP3);
end
xisup3 = [0 0];   etasup3 = [1 1];
for dir = 1:3
    homDOFs3 = findDofs3D ...
        (homDOFs3, xisup3, etasup3, dir, CP3);
end
xisup3 = [1 1];   etasup3 = [0 1];
for dir = 1:3
    homDOFs3 = findDofs3D ...
        (homDOFs3, xisup3, etasup3, dir, CP3);
end

% Inhomogeneous Dirichlet boundary conditions
inhomDOFs3 = [];
valuesInhomDOFs3 = [];

% Weak homogeneous Dirichlet boundary conditions
weakDBC3.noCnd = 0;

% Embedded cables
cables3.No = 2;
cables3.xiExtension = {[0 1] [0 1]};
cables3.etaExtension = {[0 0] [1 1]};
cables3.parameters = {parametersCable parametersCable parametersCable parametersCable};
cables3.int.type = 'default';
% cables3.int.type = 'user';
cables3.int.noGPs = 16;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Neumann boundary conditions   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% General parameter
scaling = 1e0;
FAmp = -5e2*scaling; % -5e2
n = 3;
omega = 2*pi*n/(TEnd - TStart);
snowLoad1 = @(x,y,z,t) FAmp*abs(sin(omega*t));
snowLoad2 = @(x,y,z,t) -FAmp*abs(sin(omega*t));

% Patch 1 :
% _________

% FAmp1 = loadAmplitude;
NBC1.noCnd = 1;
xib1 = [0 1];   etab1 = [0 1];   dirForce1 = 'z';
NBC1.xiLoadExtension = {xib1};
NBC1.etaLoadExtension = {etab1};
NBC1.loadAmplitude = {snowLoad1};
NBC1.loadDirection = {dirForce1};
NBC1.computeLoadVct = {'computeLoadVctAreaIGAThinStructure'};
NBC1.isFollower = false;
NBC1.isTimeDependent = true;

% Patch 2 :
% _________

% FAmp2 = - loadAmplitude;
NBC2.noCnd = 1;
xib2 = [0 1];   etab2 = [0 1];   dirForce2 = 'z';
NBC2.xiLoadExtension = {xib2};
NBC2.etaLoadExtension = {etab2};
NBC2.loadAmplitude = {snowLoad1};
NBC2.loadDirection = {dirForce2};
NBC2.computeLoadVct = {'computeLoadVctAreaIGAThinStructure'};
NBC2.isFollower = false;
NBC2.isTimeDependent = true;

% Patch 3 :
% _________

% FAmp3 = - loadAmplitude;
NBC3.noCnd = 1;
xib3 = [0 1];   etab3 = [0 1];   dirForce3 = 'z';
NBC3.xiLoadExtension = {xib3};
NBC3.etaLoadExtension = {etab3};
NBC3.loadAmplitude = {snowLoad1};
NBC3.loadDirection = {dirForce3};
NBC3.computeLoadVct = {'computeLoadVctAreaIGAThinStructure'};
NBC3.isFollower = false;
NBC3.isTimeDependent = true;

% Collect all the Neumann boundary conditions into an arra<y
NBC = {NBC1 NBC2 NBC3};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Interface parametrizations     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Patch 1 :
% _________

% Connection with patch 2:
xicoup12 = [1 1];   etacoup12 = [0 1];

% Collect all interfaces into arrays:
xicoup1 = xicoup12;
etacoup1 = etacoup12;

% Patch 2 :
% _________

% Connection with patch 1:
xicoup21 = [0 0];   etacoup21 = [0 1];

% Connection with patch 3:
xicoup23 = [1 1];   etacoup23 = [0 1];

% Collect all interfaces into arrays:
xicoup2 = [xicoup21 xicoup23];
etacoup2 = [etacoup21 etacoup23];

% Patch 3 :
% _________

% Connection with patch 2:
xicoup32 = [0 0];   etacoup32 = [0 1];

% Collect all interfaces into arrays:
xicoup3 = xicoup32;
etacoup3 = etacoup32;

% Define connections :
% ____________________

% Number of connections
noConnections = 2;

% Define connections by patch numbers
connections.No = noConnections;
connections.xiEtaCoup = zeros(noConnections, 10);
connections.xiEtaCoup(:, :) = [1 2 xicoup12 etacoup12 xicoup21 etacoup21
                               2 3 xicoup23 etacoup23 xicoup32 etacoup32];
connectionsLM = connections;
connectionsLM.lambda = struct([]);
connectionsALM = connections;
connectionsALM.lambda = struct([]);

%% Create the patches and the Lagrange Multiplier fields

% 1st patch :
% ___________

patch1 = fillUpPatch ...
    (propAnalysis, p1, Xi1, q1, Eta1, CP1, isNURBS1, parameters1, homDOFs1, ...
    inhomDOFs1, valuesInhomDOFs1, weakDBC1, cables1, NBC1, [], [], [], ...
    xicoup1, etacoup1, int1);

% 2nd patch :
% ___________

patch2 = fillUpPatch...
    (propAnalysis, p2, Xi2, q2, Eta2, CP2, isNURBS2, parameters2, homDOFs2, ...
    inhomDOFs2, valuesInhomDOFs2, weakDBC2, cables2, NBC2, [], [], [], ...
    xicoup2, etacoup2, int2);

% 3rd patch :
% ___________

patch3 = fillUpPatch ...
    (propAnalysis, p3, Xi3, q3, Eta3, CP3, isNURBS3, parameters3, homDOFs3, ...
    inhomDOFs3, valuesInhomDOFs3, weakDBC3, cables3, NBC3, [], [], [], ...
    xicoup3, etacoup3, int3);

% Collect all patches into an array :
% ___________________________________

BSplinePatches = {patch1 patch2 patch3};
noPatches = length(BSplinePatches);

%% Compute the load vector for the visualization of the reference configuration
% for counterPatches = 1:noPatches
%     BSplinePatches{counterPatches}.FGamma = ...
%         zeros(3*BSplinePatches{counterPatches}.noCPs,1);
%     for counterNBC = 1:NBC{counterPatches}.noCnd
%         funcHandle = str2func(NBC{counterPatches}.computeLoadVct{counterNBC});
%         BSplinePatches{counterPatches}.FGamma = funcHandle...
%             (BSplinePatches{counterPatches}.FGamma,...
%             BSplinePatches{counterPatches},...
%             NBC{counterPatches}.xiLoadExtension{counterNBC},...
%             NBC{counterPatches}.etaLoadExtension{counterNBC},...
%             NBC{counterPatches}.loadAmplitude{counterNBC},...
%             NBC{counterPatches}.loadDirection{counterNBC},...
%             NBC{counterPatches}.isFollower(counterNBC,1),0,...
%             BSplinePatches{counterPatches}.int,'outputEnabled');
%     end
% end

%% Plot reference configuration
% color = [.85098 .8549 .85882];
% graph.index = plot_referenceConfigurationIGAThinStructureMultipatches...
%     (BSplinePatches,connections,color,graph,'outputEnabled');

%% Create Lagrange Multipiers fields for all interfaces
if strcmp(method, 'LagrangeMultipliers') || strcmp(method, 'AugmentedLagrangeMultipliers')
    fprintf('**************************************************\n');
    fprintf('* Creating interface Lagrange Multipliers fields *\n');
    fprintf('**************************************************\n\n');
    for iConnections = 1:connections.No
        %% Get the IDs of the patches involved
        idI = connections.xiEtaCoup(iConnections, 1);
        idJ = connections.xiEtaCoup(iConnections, 2);
        fprintf('Coupling between patches %d and %d \n', idI, idJ);
        fprintf('---------------------------------- \n\n');

        %% Create a basic Lagrange Multipliers field
        pLambda = 0;
        XiLambda = [0 1];
        CPLambda(:, 4) = [1];
        isNURBSLambda = false;
        numCPs_xiLambda = length(CPLambda(:, 1, 1));
        for i = 1:numCPs_xiLambda
            if CPLambda(i, 4) ~= 1
                isNURBSLambda = true;
                break;
            end
        end

        %% Find the interface parametrization for the involved patches
        xiCoupI = connections.xiEtaCoup(iConnections, 3:4);
        etaCoupI = connections.xiEtaCoup(iConnections, 5:6);
        if xiCoupI(1, 1) ~= xiCoupI(1, 2) && etaCoupI(1, 1) == etaCoupI(1, 2)
            isOnXiI = true;
            fprintf('The coupling interface of patch %d is along xi\n', idI);
        elseif xiCoupI(1, 1) == xiCoupI(1, 2) && etaCoupI(1, 1) ~= etaCoupI(1, 2)
            isOnXiI = false;
            fprintf('The coupling interface of patch %d is along eta\n', idI);
        else
            error('Either the xi or the eta direction has to be fixed along the interface for patch %d', idI);
        end
        xiCoupJ = connections.xiEtaCoup(iConnections, 7:8);
        etaCoupJ = connections.xiEtaCoup(iConnections, 9:10);
        if xiCoupJ(1, 1) ~= xiCoupJ(1, 2) && etaCoupJ(1, 1) == etaCoupJ(1, 2)
            isOnXiJ = true;
            fprintf('The coupling interface of patch %d is along xi\n', idJ);
        elseif xiCoupJ(1, 1) == xiCoupJ(1, 2) && etaCoupJ(1, 1) ~= etaCoupJ(1, 2)
            isOnXiJ = false;
            fprintf('The coupling interface of patch %d is along eta\n', idJ);
        else
            error('Either the xi or the eta direction has to be fixed along the interface for patch %d', idJ);
        end

        %% Degree elevate the Lagrange Multipliers field
        if isOnXiI
            polOrderI = BSplinePatches{idI}.p;
        else
            polOrderI = BSplinePatches{idI}.q;
        end
        if isOnXiJ
            polOrderJ = BSplinePatches{idJ}.p;
        else
            polOrderJ = BSplinePatches{idJ}.q;
        end
        pLM = min(polOrderI, polOrderJ);
        fprintf('Degree elevating the Lagrange Multipliers field to %d\n', pLM);
        if pLM > 0
            clear pLambda XiLambda CPLambda;
            pLambda = 1;
            XiLambda = [0 0 1 1];
            CPLambda(:, 4) = [1 1];
            isNURBSLambda = false;
            numCPs_xiLambda = length(CPLambda(:, 1, 1));
            for i = 1:numCPs_xiLambda
                if CPLambda(i, 4) ~= 1
                    isNURBSLambda = true;
                    break;
                end
            end

            % Perform accordingly a p-refinement
            tpLambda = pLM;
            [XiLambda, CPLambda, pLambda] = degreeElevateBSplineCurve ...
                (pLambda, XiLambda, CPLambda, tpLambda, '');
        end

        %% Perform a knot insertion to the Lagrange Multipliers field
        if isOnXiI
            noKnotsI = length(unique(BSplinePatches{idI}.Xi)) - 2;
        else
            noKnotsI = length(unique(BSplinePatches{idI}.Eta)) - 2;
        end
        if isOnXiJ
            noKnotsJ = length(unique(BSplinePatches{idJ}.Xi)) - 2;
        else
            noKnotsJ = length(unique(BSplinePatches{idJ}.Eta)) - 2;
        end
        scaleLM = 1.0;
        noLambda = ceil(min([noKnotsI noKnotsJ])*scaleLM);
        fprintf('Uniformly inserting %d knots in the Lagrange Multipliers field\n', noLambda);
        [XiLambdaLM, CPLambdaLM] = knotRefineUniformlyBSplineCurve ...
            (noLambda, pLambda, XiLambda, CPLambda, '');
        scaleALM = 0.5;
        noLambda = ceil(min([noKnotsI noKnotsJ])*scaleALM);
        fprintf('Uniformly inserting %d knots in the augmented Lagrange Multipliers field\n', noLambda);
        [XiLambdaALM, CPLambdaALM] = knotRefineUniformlyBSplineCurve ...
            (noLambda, pLambda, XiLambda, CPLambda, '');

        %% Fill up the Lagrange Multipliers patch and add it to the array
        lambdaLM = fillUpLagrangeMultipliers ...
            (pLambda, XiLambdaLM, CPLambdaLM, isNURBSLambda);
        connectionsLM.lambda{iConnections} = lambdaLM;
        lambdaALM = fillUpLagrangeMultipliers ...
            (pLambda, XiLambdaALM, CPLambdaALM, isNURBSLambda);
        connectionsALM.lambda{iConnections} = lambdaALM;
        fprintf('\n');
    end
end

%% Output the initial geometry to be read by GiD
pathToOutput = '../../../outputGiD/isogeometricMembraneAnalysis/';
writeOutMultipatchBSplineSurface4GiD(BSplinePatches,pathToOutput,[caseName '_' meshSize '_' method]);

%% Set up the parameters and properties for each method
if strcmp(method, 'Penalty')
    % General parameters
    penaltyPrmScale = 1e0;

    % Assign the parameters for the application of weak DBC
    % -----------------------------------------------------

    for iPatches = 1:noPatches
        % Check if weak boundary conditions are to be applied
        if BSplinePatches{iPatches}.weakDBC.noCnd == 0
            continue;
        end
        
        % Assign the method name
        BSplinePatches{iPatches}.weakDBC.method = 'penalty';

        % Get the polynomial order along the Dirichlet boundary
        isOnXi = false;
        if BSplinePatches{iPatches}.weakDBC.etaExtension{1}(1, 1) == ...
                BSplinePatches{iPatches}.weakDBC.etaExtension{1}(1, 2)
            isOnXi = true;
        end
        if isOnXi
            polOrder = BSplinePatches{iPatches}.p;
        else
            polOrder = BSplinePatches{iPatches}.q;
        end

        % Assign the penalty parameter
        BSplinePatches{iPatches}.weakDBC.alpha = ...
            norm(eval(['Dm' num2str(iPatches)]))*polOrder/...
            sqrt(BSplinePatches{iPatches}.minElArea)*...
            penaltyPrmScale;
    end

    % Assign the parameters for multipatch coupling
    % ---------------------------------------------

    propCoupling.method = 'penalty';
    propCoupling.intC = intC;
    propCoupling.alphaD = zeros(connections.No, 1);
    propCoupling.alphaR = zeros(connections.No, 1);
    for iConnections = 1:connections.No
        % Get the id's of the patches
        IDPatchI = connections.xiEtaCoup(iConnections, 1);
        IDPatchJ = connections.xiEtaCoup(iConnections, 2);

        % Get the mean polynomial order between the patches
        isOnXiI = false;
        if connections.xiEtaCoup(iConnections, 5) == ...
                connections.xiEtaCoup(iConnections, 6)
            isOnXiI = true;
        end
        if isOnXiI
            polOrderI = BSplinePatches{IDPatchI}.p;
        else
            polOrderI = BSplinePatches{IDPatchI}.q;
        end
        isOnXiJ = false;
        if connections.xiEtaCoup(iConnections,9) == ...
                connections.xiEtaCoup(iConnections,10)
            isOnXiJ = true;
        end
        if isOnXiJ
            polOrderJ = BSplinePatches{IDPatchJ}.p;
        else
            polOrderJ = BSplinePatches{IDPatchJ}.q;
        end
        polOrderMean = mean([polOrderI polOrderJ]);

        % Assign the penalty parameters
        propCoupling.alphaD(iConnections, 1) = ...
            max([norm(eval(['Dm' num2str(IDPatchI)])) ...
            norm(eval(['Dm' num2str(IDPatchJ)]))])*polOrderMean/...
            sqrt(min([BSplinePatches{IDPatchI}.minElArea BSplinePatches{IDPatchJ}.minElArea]))*...
            penaltyPrmScale;
    end
elseif strcmp(method, 'LagrangeMultipliers')
    % Assign the parameters for the application of weak DBC
    % -----------------------------------------------------

    % Properties for the weak Dirichlet boundary conditions
    for iPatches = 1:noPatches
        % Check if weak boundary conditions are to be applied
        if BSplinePatches{iPatches}.weakDBC.noCnd == 0
            continue;
        end
        
        % Assign the method name
        BSplinePatches{iPatches}.weakDBC.method = 'lagrangeMultipliers';
        BSplinePatches{iPatches}.weakDBC.alpha = 0;

        % Find along which parametric line the weak Dirichlet 
        % condition is to be imposed
        isOnXi = false;
        if BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1,1) == ...
             BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1,2)
            isOnXi = true;
        end

        % Make a Lagrange Multipliers discretization
        clear pLambda XiLambda CPLambda isNURBSLambda; 

        % Find out up to which polynomial degree the Lagrange
        % Multipliers discretization needs to be increased
        if isOnXi
            polOrderPatch =  BSplinePatches{iPatches}.p;
        else
            polOrderPatch =  BSplinePatches{iPatches}.q;
        end
        pLM = polOrderPatch - 2;

        if pLM <= 0
            pLambda = 0;
            XiLambda = [0 1];
            CPLambda = zeros(1, 4);
        else
            pLambda = 1;
            XiLambda = [0 0 1 1];
            CPLambda = zeros(2, 4);

            % Peform a p-refinement
            tpLambda = polOrderPatch - pLM;
            [XiLambda, CPLambda, pLambda] = degreeElevateBSplineCurve ...
                (pLambda, XiLambda, CPLambda, tpLambda,'');
        end
        isNURBSLambda = 0;
        numCPs_xiLambda = length(CPLambda(:, 1));
        for i = 1:numCPs_xiLambda
            if CPLambda(i, 4) ~= 1
                isNURBSLambda = 1;
                break;
            end
        end

        % Perform an h-refinement
        percentage = 1.0;
        if isOnXi
            Rxi = unique(BSplinePatches{iPatches}.Xi);
        else
            Rxi = unique(BSplinePatches{iPatches}.Eta);
        end
        numXi = ceil(percentage*(length(Rxi) - 2));
        [XiLambda,CPLambda] = knotRefineUniformlyBSplineCurve...
            (numXi,pLambda,XiLambda,CPLambda,'');

        % Create the corresponding Lagrange Multipliers structure
        BSplinePatches{iPatches}.weakDBC.lambda{1} = fillUpLagrangeMultipliers...
         (pLambda, XiLambda, CPLambda, isNURBSLambda);
    end

    % Assign the parameters for multipatch coupling
    % ---------------------------------------------

    propCoupling.method = 'lagrangeMultipliers';
    connections = connectionsLM;
    propCoupling.alphaD = zeros(connections.No, 1);
    propCoupling.alphaR = zeros(connections.No, 1);
    propCoupling.intC = intC;
elseif strcmp(method, 'Mortar')
    % Assign the parameters for the application of weak DBC
    % -----------------------------------------------------

    % Properties for the weak Dirichlet boundary conditions
    for iPatches = 1:noPatches
        % Check if weak boundary conditions are to be applied
        if BSplinePatches{iPatches}.weakDBC.noCnd == 0
            continue;
        end
        
        % Assign the method name
        BSplinePatches{iPatches}.weakDBC.method = 'lagrangeMultipliers';
        BSplinePatches{iPatches}.weakDBC.alpha = 0;

        % Find along which parametric line the weak Dirichlet 
        % condition is to be imposed
        isOnXi = false;
        if BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1, 1) == ...
             BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1, 2)
            isOnXi = true;
        end

        % Make a Lagrange Multipliers discretization
        clear pLambda XiLambda CPLambda isNURBSLambda; 

        % Find out up to which polynomial degree the Lagrange
        % Multipliers discretization needs to be increased
        if isOnXi
            polOrderPatch =  BSplinePatches{iPatches}.p;
        else
            polOrderPatch =  BSplinePatches{iPatches}.q;
        end
        pLM = polOrderPatch - 2;

        if pLM <= 0
            pLambda = 0;
            XiLambda = [0 1];
            CPLambda = zeros(1, 4);
        else
            pLambda = 1;
            XiLambda = [0 0 1 1];
            CPLambda = zeros(2, 4);

            % Peform a p-refinement
            tpLambda = polOrderPatch - pLM;
            [XiLambda, CPLambda, pLambda] = degreeElevateBSplineCurve ...
                (pLambda, XiLambda, CPLambda, tpLambda, '');
        end
        isNURBSLambda = false;
        numCPs_xiLambda = length(CPLambda(:, 1));
        for i = 1:numCPs_xiLambda
            if CPLambda(i,4) ~= 1
                isNURBSLambda = true;
                break;
            end
        end

        % Perform an h-refinement
        percentage = 1.0;
        if isOnXi
            Rxi = unique(BSplinePatches{iPatches}.Xi);
        else
            Rxi = unique(BSplinePatches{iPatches}.Eta);
        end
        numXi = ceil(percentage*(length(Rxi) - 2));
        [XiLambda, CPLambda] = knotRefineUniformlyBSplineCurve ...
            (numXi, pLambda, XiLambda, CPLambda, '');

        % Create the corresponding Lagrange Multipliers structure
        BSplinePatches{iPatches}.weakDBC.lambda{1} = fillUpLagrangeMultipliers...
         (pLambda,XiLambda,CPLambda,isNURBSLambda);
    end

    % Assign the parameters for multipatch coupling
    % ---------------------------------------------

    propCoupling.method = 'mortar';
    propCoupling.isSlaveSideCoarser = false;
    propCoupling.computeRearrangedProblemMtrcs = @computeRearrangedProblemMtrcs4MortarIGAMembrane;
    propCoupling.intC = intC;
elseif strcmp(method, 'AugmentedLagrangeMultipliers')
    % Assign the parameters for the application of weak DBC
    % -----------------------------------------------------
    
    % Scaling factor for the augmented Lagrange Multipliers method
    scalePenaltyFactorALM = 1e-2;
    
    for iPatches = 1:noPatches
        % Check if weak boundary conditions are to be applied
        if BSplinePatches{iPatches}.weakDBC.noCnd == 0
            continue;
        end
        
        % Assign the method name
        BSplinePatches{iPatches}.weakDBC.method = 'lagrangeMultipliers';

        % Find along which parametric line the weak Dirichlet
        % condition is to be imposed
        isOnXi = false;
        if BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1, 1) == ...
             BSplinePatches{iPatches}.weakDBC.xiExtension{1}(1, 2)
            isOnXi = true;
        end

        % Find out up to which polynomial degree the Lagrange
        % Multipliers discretization needs to be increased
        if isOnXi
            polOrderPatch = BSplinePatches{iPatches}.p;
        else
            polOrderPatch = BSplinePatches{iPatches}.q;
        end
        pLM = polOrderPatch - 2; %polOrderPatch - 2 has worked well

        % Assign the penalty parameter
        BSplinePatches{iPatches}.weakDBC.alpha = ...
            norm(eval(['Dm' num2str(iPatches)]))*polOrderPatch/...
            sqrt(BSplinePatches{iPatches}.minElArea)*scalePenaltyFactorALM;

        % Make a Lagrange Multipliers discretization
        clear pLambda XiLambda CPLambda isNURBSLambda; 
        if pLM <= 0
            pLambda = 0;
            XiLambda = [0 1];
            CPLambda = zeros(1,4);
        else
            pLambda = 1;
            XiLambda = [0 0 1 1];
            CPLambda = zeros(2,4);

            % Peform a p-refinement
            tpLambda = polOrderPatch - pLM;
            [XiLambda,CPLambda,pLambda] = degreeElevateBSplineCurve...
                (pLambda,XiLambda,CPLambda,tpLambda,'');
        end
        isNURBSLambda = false;
        numCPs_xiLambda = length(CPLambda(:,1));
        for i = 1:numCPs_xiLambda
            if CPLambda(i,4) ~= 1
                isNURBSLambda = true;
                break;
            end
        end

        % Perform an h-refinement
        percentage = .5;
        if isOnXi
            Rxi = unique(BSplinePatches{iPatches}.Xi);
        else
            Rxi = unique(BSplinePatches{iPatches}.Eta);
        end
        numXi = ceil(percentage*(length(Rxi) - 2));
        [XiLambda, CPLambda] = knotRefineUniformlyBSplineCurve ...
            (numXi, pLambda, XiLambda, CPLambda, '');

        % Create the corresponding Lagrange Multipliers structure
        BSplinePatches{iPatches}.weakDBC.lambda{1} = fillUpLagrangeMultipliers ...
            (pLambda, XiLambda, CPLambda, isNURBSLambda);
    end
    
    % Assign the parameters for multipatch coupling
    % ---------------------------------------------

    propCoupling.method = 'lagrangeMultipliers';
    connections = connectionsALM;
    propCoupling.intC = intC;
    propCoupling.alphaD = zeros(connections.No, 1);
    propCoupling.alphaR = zeros(connections.No, 1);
    for iConnections = 1:connections.No
        % Get the Patch IDs
        IDPatchI = connections.xiEtaCoup(iConnections, 1);
        IDPatchJ = connections.xiEtaCoup(iConnections, 2);

        % Get the mean polynomial order between the patches
        isOnXiI = false;
        if connections.xiEtaCoup(iConnections, 5) == ...
                connections.xiEtaCoup(iConnections, 6)
            isOnXiI = true;
        end
        if isOnXiI
            polOrderI = BSplinePatches{IDPatchI}.p;
        else
            polOrderI = BSplinePatches{IDPatchI}.q;
        end
        isOnXiJ = false;
        if connections.xiEtaCoup(iConnections, 9) == ...
                connections.xiEtaCoup(iConnections, 10)
            isOnXiJ = true;
        end
        if isOnXiJ
            polOrderJ = BSplinePatches{IDPatchJ}.p;
        else
            polOrderJ = BSplinePatches{IDPatchJ}.q;
        end
        polOrderMean = mean([polOrderI polOrderJ]);

        % Assign the penalty parameter
        propCoupling.alphaD(iConnections, 1) = ...
            max([norm(eval(['Dm' num2str(IDPatchI)])) ...
            norm(eval(['Dm' num2str(IDPatchJ)]))])*polOrderMean/...
            sqrt(min([BSplinePatches{IDPatchI}.minElArea BSplinePatches{IDPatchJ}.minElArea]))*...
            scalePenaltyFactorALM;
    end
elseif strcmp(method, 'Nitsche')
    % Assign the parameters for the application of weak DBC
    % -----------------------------------------------------

    % Properties for the weak Dirichlet boundary conditions
    for iPatches = 1:noPatches
        % Check if weak boundary conditions are to be applied
        if BSplinePatches{iPatches}.weakDBC.noCnd == 0
            continue;
        end
        
        % Assign the method
        BSplinePatches{iPatches}.weakDBC.method = 'nitsche';
        
        % Assign the estimation of the stabilization parameter
        BSplinePatches{iPatches}.weakDBC.estimationStabilPrm = true;
    end

    % Assign the parameters for multipatch coupling
    % ---------------------------------------------

    propCoupling.method = 'nitsche';
    propCoupling.estimationStabilPrm = true;
    propCoupling.gammaTilde = .5;
    propCoupling.intC = intC;
end

%% Nonlinear analysis properties

% number of load steps for the non-linear analysis
propNLinearAnalysis.method = 'newtonRapshon';

% number of load steps for the non-linear analysis
propNLinearAnalysis.noLoadSteps = 1;

% Assign a tolerance for the Newton iterations
propNLinearAnalysis.eps = 1e-4;

% Assign the maximum number of iterations
propNLinearAnalysis.maxIter = 100;

%% Determine the Rayleigh damping coefficients
% alphaR = 17.4670;
% betaR = 1.8875e-04;

% Assign two significant eigenfrequencies
freqI = 4.656892249680108;
freqJ = 8.953736617642974;

% Assign the corresponding logarithmic decrements
zetaI = .1;
zetaJ = .1;

% Compute the corresponding circular frequencies
omegaI = 2*pi*freqI;
omegaJ = 2*pi*freqJ;

% Compute the Rayleigh damping parameters
detM = omegaJ/omegaI - omegaI/omegaJ;
coef = 2/detM;
alphaR = coef*(omegaJ*zetaI - omegaI*zetaJ);
betaR = coef*(-1/omegaJ*zetaI + 1/omegaI*zetaJ);

%% Transient analysis properties

% Time dependence
propStrDynamics.timeDependence = 'transient';

% Time integration scheme
% propStrDynamics.method = 'explicitEuler';
propStrDynamics.method = 'bossak';
propStrDynamics.alphaB = 0; % -.1
propStrDynamics.betaB = .25; % .5
propStrDynamics.gammaB = .5; % .6
propStrDynamics.damping.method = 'rayleigh';
propStrDynamics.damping.computeDampMtx = @computeDampMtxRayleigh;
propStrDynamics.damping.alpha = alphaR;
propStrDynamics.damping.beta = betaR;

% Function handle to the computation of the matrices for the defined time
% integration scheme
if strcmp(propStrDynamics.method, 'explicitEuler')
    propStrDynamics.computeProblemMtrcsTransient = ...
        @computeProblemMtrcsExplicitEuler;
elseif strcmp(propStrDynamics.method, 'bossak')
    propStrDynamics.computeProblemMtrcsTransient = ...
        @computeProblemMtrcsBossak;
else
    error('Choose a time integration scheme');
end

% Function handle to the update of the discrete fields for the defined time
% integration scheme
if strcmp(propStrDynamics.method, 'explicitEuler')
    propStrDynamics.computeUpdatedVct = ...
        @computeBETITransientUpdatedVctAccelerationField;
elseif strcmp(propStrDynamics.method, 'bossak')
    propStrDynamics.computeUpdatedVct = ...
        @computeBossakTransientUpdatedVctAccelerationField;
else
    error('Choose a time integration scheme');
end

% Function handle to the initial conditions computation
computeInitCnds = @computeNullInitCndsIGAThinStructure;
% computeInitCnds = @computeRestartInitCndsIGAGiD;

% Starting time of the simulation
propStrDynamics.TStart = TStart;

% End time of the simulation
propStrDynamics.TEnd = TEnd;

% Number of time steps
if strcmp(meshSize, 'Coarse')
   timeStepScaling = 500; % 16 
elseif strcmp(meshSize, 'Fine')
    timeStepScaling = 500; % 16
end
propStrDynamics.noTimeSteps = timeStepScaling*(propStrDynamics.TEnd-propStrDynamics.TStart);

% Time step size
propStrDynamics.dt = (propStrDynamics.TEnd - propStrDynamics.TStart)/propStrDynamics.noTimeSteps;
str_timeStep = num2str(propStrDynamics.dt);

%% Solve the transient problem
% writeOutputLM = 'undefined';
propOutputNitsche = propOutput;
computeInitCnd = @computeNullInitCndsIGAThinStructure;
[dHatHistory,resHistory,BSplinePatches,propCoupling,~] = ...
 solve_DDMIGAMembraneMultipatchesNLinearTransient...
 (BSplinePatches, connections, computeInitCnd, propCoupling, ...
 propNLinearAnalysis, propStrDynamics, propPostproc, solve_LinearSystem, ...
 propOutputNitsche, pathToOutput, [caseName '_' meshSize '_' method], ...
 propEmpireCoSimulation, 'outputEnabled');
% graph.index = plot_postprocIGAMembraneMultipatchesNLinear...
%     (BSplinePatchesNitsche,dHatHistoryNitsche(:,end),graph,'outputEnabled');
save(['./data_TransientAnalysisPool/' 'data_' caseName '_' meshSize '_' method '_dt_' str_timeStep '.mat']);
return;

%% Postprocessing

% On the postprocessing
graph.postprocConfig = 'current'; % 'reference','current','referenceCurrent'
graph.resultant = 'force'; % 'displacement','strain','curvature','force','moment','shearForce'
graph.component = '1Principal'; % 'x','y','z','2norm','1','2','12','1Principal','2Principal'

% Visualize the deformed configuration with the selected resultants
% scaling = 1;
% t = .81; % .13;
scaling = 10; % 10
t = .36; % .13;
noTimeStep = int64((t - propStrDynamics.TStart)/propStrDynamics.dt + 2);
graph.index = graph.index + 1;
graph.index = plot_postprocIGAMembraneMultipatchesNLinear...
    (BSplinePatches,scaling*dHatHistory(:,noTimeStep),graph,'outputEnabled');
az = 260;
el = 30;
view(az,el);
camlight(30,70);
lighting phong;
axis off;
title('');
h = colorbar;
limitsColorbar = get(h,'Limits');
minimum = 0.131119006275351e4; % 0.331119006275351e4
% maximum = 1.347001710567534e4
caxis([minimum limitsColorbar(1,2)]);

% Write the geometry for GiD
BSplinePatchesMod = BSplinePatches;
BSplinePatchesMod = computeUpdatedGeometryIGAThinStructureMultipatches...
    (BSplinePatches,scaling*dHatHistory(:,noTimeStep));
for iPatches = 1:noPatches
    BSplinePatchesMod{iPatches}.CP = BSplinePatchesMod{iPatches}.CPd;
end
pathToOutput = '../../../outputGiD/isogeometricMembraneAnalysis/';
writeOutMultipatchBSplineSurface4GiD(BSplinePatchesMod,pathToOutput,[caseName '_' meshSize '_' method]);

return;

% Plot the stabilization parameter over the time
timeAxis = zeros(propStrDynamics.noTimeSteps + 1,1);
for iTimeSteps = 1:propStrDynamics.noTimeSteps
    timeAxis(iTimeSteps + 1,1) = iTimeSteps*propStrDynamics.dt;
end
semilogy(timeAxis,propCoupling.automaticStabilization(:,:));
 
% % Plot the load amplitude
% loadAmp = zeros(propStrDynamics.noTimeSteps + 1,1)
% t = 0;
% for iTimeSteps = 1:propStrDynamics.noTimeSteps + 1
%     loadAmp(iTimeSteps,1) = snowLoad1(0,0,0,t);
%     t = t + propStrDynamics.dt;
% end
% plot(timeAxis,loadAmp);

%% END