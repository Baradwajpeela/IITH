import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy

# Equation of planes:
# 2x - y +2z = 2
# x- 2y + z = -4
# x + y + kz = 4

def determinant(A):
    return (A[0,0]*A[1,1]*A[2,2] + A[0,1]*A[1,2]*A[2,0]+A[0,2]*A[1,0]*A[2,1]-A[2,0] *A[1,1]*A[0,2]-A[2,1]*A[1,2]*A[0,0]-A[2,2]*A[1,0] *A[0,1])

# For plotting 3D figures
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Solving for the value of k
k = sympy.Symbol('k')

# Giving the values of normal vectors in terms of matrices
# Giving any point lying on the plane so that an unique given plane can be plotted
point1  = np.array([0, 0, 1])
normal1 = np.array([2, -1, 2])

point2  = np.array([0, 0, -4])
normal2 = np.array([1, -2, 1])

point3  = np.array([2, 2, 0])
normal3 = np.array([1, 1, k])

point = np.vstack((point1,point2,point3))
normal = np.vstack((normal1,normal2,normal3))
d = np.array([2,-4,4])

# Solving for k by making the determinant of coefficient matrix equal to zero
det = determinant(normal)
print(sympy.solve(det))

len = 11
lam = np.linspace(0,1,len)

# Initialising the values of A,B,C,D,E and F such that A and B are lying on the first and second plane , C and D on second and third plane and so on.
A = np.array([8/3, 10/3, 0])
B = np.array([5/3,10/3 ,1 ])

C = np.array([4/3, 8/3, 0])
D = np.array([1-(2*k/3), 3-(k/3), 1])

E = np.array([2, 2, 0])
F = np.array([(4-k)/3, (8-2*k)/3, 1])

# Verification for more values of k
# Substituting the values of k
l = int(input('Enter the value of k: '))
D = np.array([3/5, 3-(l/3), 1])
F = np.array([(4-l)/3, (8-2*l)/3, 1])
normal[2,2] = l
det = determinant(normal)

# Getting the point of intersection(if it exists) to fit in the plane.
if det !=0 :
    x = d/det
    print(x)

    A = np.array([x[0] - 15, x[1], x[2] + 15])
    B = np.array([x[0] + 15, x[1], x[2] - 15])

    C = np.array([x[0] + 5 * (2 * l + 1), x[1] - 5 * (1 - l), x[2] - 15])
    D = np.array([x[0] - 5 * (2 * l + 1), x[1] + 5 * (1 - l), x[2] + 15])

    E = np.array([x[0] - 5 * (2 + l), x[1] - 5 * (2 * l - 2), x[2] + 15])
    F = np.array([x[0] + 5 * (2 + l), x[1] + 5 * (2 * l - 2), x[2] - 15])

# If there exists no solution Values of all point will be same
else:
    print('Their exist no solution')

# Initializing all the lines AB,CD and EF.
AB = np.zeros((3,len))
CD = np.zeros((3,len))
EF = np.zeros((3,len))

# Writing all the points lying on the lines AB,CD and EF
for i in range(len):
    AB[:,i] = A + lam[i]*(B-A)
    CD[:,i] = C + lam[i]*(D-C)
    EF[:,i] = E + lam[i]*(F-E)

# Plotting all the lines and then marking the end points of the line
ax.plot(AB[0,:], AB[1,:],AB[2,:] , label ='AB')
ax.text(A[0], A[1],A[2], 'A')
ax.text(B[0], B[1],B[2], 'B')
ax.plot(CD[0,:], CD[1,:],CD[2,:] , label ='CD')
ax.text(C[0], C[1],C[2], 'C')
ax.text(D[0], D[1],D[2], 'D')
ax.plot(EF[0,:], EF[1,:],EF[2,:] , label ='EF')
ax.text(E[0], E[1],E[2], 'E')
ax.text(F[0], F[1],F[2], 'F')

# Plotting the points A,B,C,D,E and F
ax.scatter(A[0], A[1], A[2],'o')
ax.scatter(B[0], B[1], B[2],'o')
ax.scatter(C[0], C[1], C[2],'o')
ax.scatter(D[0], D[1], D[2],'o')
ax.scatter(E[0], E[1], E[2],'o')
ax.scatter(F[0], F[1], F[2],'o')

# Plotting the point of intersection of all lines
if det != 0:
    ax.scatter(x[0], x[1], x[2], 'o')
    ax.text(x[0]*(1+0.1), x[1]*(1+0.1), x[2]*(1+0.1), 'I')

# Labelling the x,y and z axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.legend(loc = 'best')

# For showing plot
plt.show()