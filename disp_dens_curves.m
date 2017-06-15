%%
%EMPA 2017
%Felipe Diaz
%This function analyses the density and displacement curves of the bone
%samples. In order to succesfully run the program, check that the folders
%coincide with the syntax used in this program. 
%INPUTS:
%name: (string) is the name of the sample.
%Nz: (integer) is the Z-dimension of the image stack to be analysed.
%sn: (integer) is the sample number (numbers after letter of the sample name)

function [] = disp_dens_curves(name, Nz, sn)

%% Data reading
sn = num2str(sn);
a = 'force/CaseNr ';
b = [a sn];
row = 500;
col = 500;
dx = 0.0041;
fin = fopen(strcat('Bones/', name, '_500x500x', num2str(Nz), '.raw'),'r');
I = fread(fin, row * col * Nz, 'single=>single'); 
I = reshape(I, row, col, Nz);

switch name
    case{'SC3'}
        M = dlmread(strcat(b, '/KalotteNo', num2str(sn), '.csv'), ';', 3, 0);
    otherwise
        M = dlmread(strcat(b, '/KalotteNo', num2str(sn), '.csv'), ';', 2, 0);
end
    

st_force = M(:, 4);
st_weg = M(:, 5);

%% Density calculation

%(First approach)

% mval = mean(mean(I, 1), 2);
% mval = reshape(mval, 1, Nz);
% rho = 4226.9887 * mval - 760.0887; 
% rho = rho(rho > 0);
% x_vec = 0:dx:((length(rho) - 1)* dx);
% x_vec2 = max(x_vec) - (st_weg / 10);
% inte = trapz(rho);
% norm = max(x_vec);
% int_norm = inte / norm;

%(Second approach)

mval = mean(mean(I, 1), 2);
mval = reshape(mval, 1, Nz);
rho = 4226.9887 * mval - 760.0887; 
rho = rho(rho > 0);
x_vec = 0:dx:((length(rho) - 1)* dx);
x_vec2 = max(x_vec) - (st_weg / 10);
x_vec = x_vec(x_vec > min(x_vec2));
rho = rho(end:-1:end -length(x_vec) + 1);
rho = rho(end:-1:1);
inte = trapz(rho);
norm = max(x_vec);
int_norm = inte / norm;
[~,m,~] = regression(st_weg, st_force, 'one');

%% Plotting 

figure()
plot(x_vec, rho, 'b');
legend(strcat('Norm. Integral=', num2str(int_norm)), 'Location', 'Southwest');
xlabel('Depth (cm)'); ylabel('Density (mg HA/cm ^3)', 'Color', 'b');
ax1 = gca;
ax1_pos = ax1.Position;
ax2 = axes('Position',ax1_pos, 'XAxisLocation', 'top', 'YAxisLocation', 'right', 'Color','none');
line(x_vec2,st_force,'Parent',ax2, 'Color', 'k');
ylabel('Force (N)');
ax1.XLim = [0 2]; ax1.YLim = [0 900];
ax2.XLim = [0 2]; ax2.YLim = [0 1200];
legend(strcat('Slope=', num2str(m)));
grid on;
