import GPyOpt
def myf(x):
  return x**2

bounds=[{'name':'var_1','type':'continuous','domain':(-1,1)}]
# 变量名字，连续变量，定义区间是-1到1
max_iter=15
# 最大迭代次数
myProblem=GPyOpt.methods.BayesianOptimization(myf,bounds)
#用贝叶适优化来求解这个函数，函数的约束条件是bounds
myProblem.run_optimization(max_iter)
#开始求解
print(myProblem.x_opt)
#打印最优解对应的x为-0.00103
print(myProblem.fx_opt)
#打印最优解对应d的函数值为0.0004
