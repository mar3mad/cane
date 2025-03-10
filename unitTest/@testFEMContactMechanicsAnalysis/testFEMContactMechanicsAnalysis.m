classdef testFEMContactMechanicsAnalysis < matlab.unittest.TestCase
%% Licensing
%
% License:         BSD License
%                  cane Multiphysics default license: cane/license.txt
%
% Main authors:    Andreas Apostolatos
%
%% Class definition
%
% Test suites for the finite element formulation of the frictionless 
% Signorini contact problem in 2D.
%
%% Method definitions
methods (Test)
    testFrictionlessSignoriniContactBridge2D(testCase)
    testFrictionlessSignoriniContactCantileverBeam2D(testCase)
    testFrictionlessSignoriniContactWedge2D(testCase)
    testFrictionlessSignoriniContactHertz2D(testCase)
end
    
end
