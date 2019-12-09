% This file is part of eastereig, a library to locate exceptional points
% and to reconstruct eigenvalues loci.

% Eastereig is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% Eastereig is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.

% You should have received a copy of the GNU General Public License
% along with Eastereig.  If not, see <https://www.gnu.org/licenses/>.

function out=PlotRiemann(beta,Pr,Pi,Type,N,label)
% Plot Riemman surfaces of the selected eigenvalues.
%
% To avoid discontinuity of beta between the different Riemann surfaces, 
% the real or imaginary part of the values are sorted and the delaunay tessalation 
% are used to create possibility dicontinuous surfaces
%
%Parameters
%-----------
% beta : is a cell of dim size(Pr) x nb eigenval. The Riemman surface to plot
% Pr, Pi : are array that contains the repectivelly the real and the imaginary part of the parameter where the beta(i,j) are computed
% Type : {'Re', 'Im'}, default: 'Re'; Specify which part of the complex parameter is plotted          
% N : Maximum number of eigenvalues to be plotted    
% Label : default 'p'; Variable name for axis labels      

% define default parameters
if nargin ==3
    Type  = 'Re'; % 'Im'
    N= 3;
    label={'$p$','$\beta$'};
end

if nargin ==4
    N = 3;
    label={'$p$','$\beta$'};
end

if  ~iscell(beta)
    fprintf('> beta is not a cell, assume 3D matrix for conversion...\n')
    beta=num2cell(beta,3);    
end

% set figure style
fontname = 'Times New Roman';
set(0,'defaultaxesfontname',fontname);
set(0,'defaulttextfontname',fontname);
fontsize = 12;
set(0,'defaultaxesfontsize',fontsize);
set(0,'defaulttextfontsize',fontsize);

% init
betaplot = cell(1,min([length(beta{1,1}(:)), N]));

for mode=1:length(betaplot)
    betaplot{mode} = zeros(size(Pr));
end

% sort the data 
for i=1:size(Pr,1)
    for j=1:size(Pr,2)
        if strcmp(Type,'Re')
            % la plus faible Re
            [B,I]=sort(real(beta{i,j}),'descend');
        else
            % la plus faible Im
            [B,I]=sort(imag(beta{i,j}),'ascend');
        end
        beta{i,j}=B;
        for mode=1:length(betaplot)%min([length(B), N])
            
            betaplot{mode}(i,j) = B(mode);            
        end
    end
end

% delaunay
y= Pi(:);
x= Pr(:);
tri = delaunay(x,y);
% How many triangles are there?
[r,c] = size(tri);
disp(r)

% Plot it with TRISURF
figure
% hold on
for i=1:length(betaplot)    
    h = trisurf(tri, x, y, betaplot{i}(:),'LineStyle','-','EdgeColor','k','FaceAlpha',0.7,'EdgeAlpha',0.7)%,'FaceColor','none')%,'Marker','.','MarkerEdgeColor','blue');
    hold on
end

if strcmp(Type,'Re')
    zlabel(['Re  ' label{2}],'Interpreter', 'latex')
else
    zlabel(['Im  ' label{2}],'Interpreter', 'latex')
end
xlabel(['Re  ' label{1}],'Interpreter', 'latex')
ylabel(['Im  ' label{1}],'Interpreter', 'latex')
view(-49.5000,24)

% % plot real line(beta,Pr,Pi,Type,N)
% plot3(Pr(1,:),Pi(1,:),betaplot{1}(1,:),'linewidth',3.5,'color',[0 128 0]/255)
% plot3(Pr(1,:),Pi(1,:),betaplot{2}(1,:),'linewidth',3.5,'color','b')

out=1;
