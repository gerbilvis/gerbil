#!/bin/bash
gnuplot -persist << EOF 
# set output "modifiedGauss.png"
set title "Gauss" 
set xlabel "x" 
set ylabel "p(x)" 
set zlabel "" 
unset logscale x 
unset logscale y 
unset logscale z

m_iter = 40000
ten = m_iter/10.0
# ten = ten
# set xtics 100
set xrange [ 0 : m_iter]
set yrange [ 0 : 1.2 ]
# 
set title "Radius 8" 
depthMax = m_iter/2
mx = 0
# sigma = depthMax/2
sigma = depthMax/6.0
sigma = ten/2.0
# sigma = 1
learning_start = 0.1
learning_end = 0.001
# g(x)=exp( - (x-mx)*(x-mx) / (2.0 * sigma * sigma)) 
# g(x)= (-1/(1+exp(-1/ten*(x-ten))) +1) /
# g(x)= (x<= ten) ? (-1/(1+exp(-1/2000.0*(x-0))) +1) : (-1/(1+exp(-1/ten*(x-ten))) +1)
g(x) = (x<= ten) ? exp(- (abs(x-0)**2/(2*sigma*sigma))) : (exp(- (abs(ten)**2/(2*sigma*sigma))) *(1-x/m_iter)) * ((learning_end/(exp(- (abs(ten)**2/(2*sigma*sigma)))*(1-x/m_iter)) )**(x/ m_iter))
# g(x) =  exp(-0.0001*x)
# g(x)= tanh(x)
# g(x) =  exp(- (abs(x-0)**2/(2*sigma*sigma)))
#  g(x)= (-1/(1+exp(-1/ten*(x-ten))) +1) /(-1/(1+exp(-1/ten*(0-ten))) +1)
plot [:] [:]g(x)  with lines 



set xrange [ 0 : m_iter ]
set yrange [ 0 : 1.0 ]


radius_start = 8.0
radius_end = 0.5



# a(x)= (x< 1000) ? 0.9*(1-x/m_iter) : (0.9*(1-x/m_iter)) * ((learning_end/(0.9*(1-x/m_iter)) )**(x/ m_iter))

# a(x)=	floor(radius_start * ( (1.0/radius_start)**(x/m_iter) ))
# b(x)= floor(radius_start * ( (0.5/radius_start)**(x/m_iter) ))
# c(x)= floor(radius_start * ( (0.2/radius_start)**(x/m_iter) ))
# d(x)= (learning_start * ( (learning_end/learning_start)**(x/m_iter) ))
# e(x)= floor(radius_start * ( (0.01/radius_start)**(x/m_iter) ))

# plot [:] [:] d(x) with lines ls 1 
# ,\
# b(x) with lines ls 2 ,\
# c(x) with lines ls 3 ,\
# d(x) with lines ls 4 ,\
# e(x) with lines ls 5 

EOF