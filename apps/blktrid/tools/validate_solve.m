% Written by Endre Laszlo, James Whittle and Catherine Hastings, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 
% Solve Ax=b

SYS_LEN = 256;
BLK_DIM = 3;

A = zeros(SYS_LEN,SYS_LEN);
for i=1:SYS_LEN
  for j=1:SYS_LEN
    if(i==j)                                      A(i,j) = 4.0;
    elseif(abs(i-j)==1)                           A(i,j) = 0.001; 
    elseif(j == (i-BLK_DIM) || j == (i+BLK_DIM) ) A(i,j) = 0.001; 
    else                                          A(i,j) = 0.0; 
    end;
  end
end

b = ones(SYS_LEN,1);

x = A\b

cond(A)
eigs(A)