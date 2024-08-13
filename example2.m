%% non-uniform hypergraph
clear all
clc
syms t x1(t) x2(t) x3(t) x4(t) x5(t) x6(t) x7(t)

% number of nodes  
n = 7;

% 2-uniform graph
A1 = sym(zeros(n,n));

% 3-uniform hypergraph
[i1 i2 i3] = ndgrid(1:n,1:n,1:n);
k = arrayfun(@(j1)sym(sprintf('0',i1(j1),i2(j1),i3(j1))),...
              1:numel(i1),'un',0);
A2 = reshape([k{:}],n,n,[]);

%% Construct temporal hypergraph A1 and A2
%% Case 1:
e = [2 5];
e = perms(e);
A1(sub2ind(size(A1),e(:,1),e(:,2))) = t;


e = [3 6];
e = perms(e);
A1(sub2ind(size(A1),e(:,1),e(:,2))) = t;

e = [1 3 5];
e = perms(e);
A2(sub2ind(size(A2),e(:,1),e(:,2),e(:,3))) = t;

e = [2 4 7];
e = perms(e);
A2(sub2ind(size(A2),e(:,1),e(:,2),e(:,3))) = t;


%% Case 2: 

e = [3 7];
e = perms(e);
A1(sub2ind(size(A1),e(:,1),e(:,2))) = t;

e = [4 7];
e = perms(e);
A1(sub2ind(size(A1),e(:,1),e(:,2))) = t;

e = [5 6];
e = perms(e);
A1(sub2ind(size(A1),e(:,1),e(:,2))) = t;

e = [1 3 5];
e = perms(e);
A2(sub2ind(size(A2),e(:,1),e(:,2),e(:,3))) = t;

e = [1 2 5];
e = perms(e);
A2(sub2ind(size(A2),e(:,1),e(:,2),e(:,3))) = t;

e = [2 5 6];
e = perms(e);
A2(sub2ind(size(A2),e(:,1),e(:,2),e(:,3))) = t;


%% Define symbolic x\in R^n B \in R^{n*m}

B = transpose([1,1,0,1,0,0,0]);
% B = transpose([5.*t,2.*t,3.*t,t,2.*t,4.*t,t]);
X = transpose([x1(t),x2(t),x3(t),x4(t),x5(t),x6(t),x7(t)]);

%% Controllability matrix M_i i = 1,2,3
% 1.homogeneous case
tic

M0 = B;
M1 = Mcompute(A1,A2,M0,B,X);
M2 = Mcompute(A1,A2,M1,B,X);
M3 = Mcompute(A1,A2,M2,B,X);
M4 = Mcompute(A1,A2,M3,B,X);
M5 = Mcompute(A1,A2,M4,B,X);
M6 = Mcompute(A1,A2,M5,B,X);

CM = [M0 M1 M2 M3 M4 M5 M6];

toc
%% Rank condition --full rank for weakly controlbility

rnk = rank(CM)

%% Symbolic functions
% Computing M in homogeneous case 
function Mnext = Mcompute(A1,A2,M,B,X)
syms t x1(t) x2(t) x3(t) x4(t) x5(t) x6(t) x7(t)

Mnext = A1*M+2.*tensorprod(A2,X)*M-diff(M,t)-jacobian(M,X)*B;

u = 1;

% Dynamics
dX = A1*X + tensorprodsqrt(A2,X) + u.*B;
n = length(X); 

dx1 = dX(1);
dx2 = dX(2);
dx3 = dX(3);
dx4 = dX(4);
dx5 = dX(5);
dx6 = dX(6);
dx7 = dX(7);

Mnext = subs(Mnext, diff(x1(t), t), dx1);
Mnext = subs(Mnext, diff(x2(t), t), dx2);
Mnext = subs(Mnext, diff(x3(t), t), dx3);
Mnext = subs(Mnext, diff(x4(t), t), dx4);
Mnext = subs(Mnext, diff(x5(t), t), dx5);
Mnext = subs(Mnext, diff(x6(t), t), dx6);
Mnext = subs(Mnext, diff(x7(t), t), dx7);

end

% Symbolic tensor product for X^2
function C = tensorprod(A,X)
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