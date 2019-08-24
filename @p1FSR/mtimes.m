function res = mtimes(a,b)

if a.adjoint
    res=pinv(a.keymatrix)*pinv(a.P_matrix)*b;


else
    res=a.P_matrix*a.keymatrix*b;

end



    
