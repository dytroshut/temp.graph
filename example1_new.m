clear all
clc

% number of nodes  
n = 4;

syms t
syms x(t) [4 1]

[i1 i2 i3] = ndgrid(1:n,1:n,1:n);
k = arrayfun(@(j1)sym(sprintf('0',i1(j1),i2(j1),i3(j1))),...
              1:numel(i1),'un',0);
A = reshape([k{:}],n,n,[]);
%% Construct temporal hypergraph A
% Case 1
% e = [1 3 4];
% e1 = perms(e);
% A(sub2ind(size(A),e1(:,1),e1(:,2),e1(:,3))) = 2.*t;
% 
% e = [1 2 4]; 
% e2 = perms(e);
% A(sub2ind(size(A),e2(:,1),e2(:,2),e2(:,3))) = -t;

% % e = [2 3 4];
% % e3 = perms(e);
% % A(sub2ind(size(A),e3(:,1),e3(:,2),e3(:,3))) = t;

%% Case 2
e = [1 3 4];
e1 = perms(e);
A(sub2ind(size(A),e1(1:2,1),e1(1:2,2),e1(1:2,3))) = 2*t;

e = [1 2 4]; 
e2 = perms(e);
A(sub2ind(size(A),e2(5:6,1),e2(5:6,2),e2(5:6,3))) = t.^2;

e = [2 3 4];
e3 = perms(e);
A(sub2ind(size(A),e3(3:4,1),e3(3:4,2),e3(3:4,3))) = 1/t;


%% Define symbolic x\in R^n B \in R^{n*m}
% B = transpose([5.*t 2.*t 3.*t t]);
% B = transpose([1/t 1/t 1/t 1/t]);
B = transpose([2*t 1./t t t.^2]);
X = transpose([x1(t), x2(t), x3(t), x4(t)]);
%% Controllability matrix M_i i = 1,2,3
% 1.homogeneous case
M0 = B;
M1 = Mcompute(A,M0,B,X);
M2 = Mcompute(A,M1,B,X);
M3 = Mcompute(A,M2,B,X);

CM = [M0 M1 M2 M3]
%% Rank condition --full rank for weakly controlbility
rnk = rank(CM)

%% Symbolic functions

% Computing M in homogeneous case 
function Mnext = Mcompute(A,M,B,X)
syms t x1(t) x2(t) x3(t) x4(t) 

Mnext = 2.*tensorprodd(A,X)*M-diff(M,t)-jacobian(M,X)*B - tensorprodd(A,X.^2)*diff(M,X);

u = 1;

% Dynamics
dX = tensorprodsqrt(A,X) + u.*B;
n = length(X); 

dx1 = dX(1);
dx2 = dX(2);
dx3 = dX(3);
dx4 = dX(4);
Mnext = subs(Mnext, diff(x1(t), t), dx1);
Mnext = subs(Mnext, diff(x2(t), t), dx2);
Mnext = subs(Mnext, diff(x3(t), t), dx3);
Mnext = subs(Mnext, diff(x4(t), t), dx4);
end

% Symbolic tensor product for X^2

function C = tensorprodd(A,X)
syms x t
n = length(X); 

for i = 1:n  
C(:,i) = A(:,:,i)*X;
end

end

% Symbolic tensor product for X^2
function C = tensorprodsqrt(A,X)
syms x t
n = length(X); 

for i = 1:n  
C(:,i) = A(:,:,i)*X;
end

C = C*X;
end
