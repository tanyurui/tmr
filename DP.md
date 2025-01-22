# DP

### 23421: 小偷背包 

dp, http://cs101.openjudge.cn/practice/23421

思路：典

代码：（用类似题目替代了）

```python
t,m=map(int,input().split())
grass=[list(map(int,input().split())) for _ in range(m)]
dp=[0]*(t+1)
for time,value in grass:
    for j in range(t,time-1,-1):#倒序保证只拿一次
        dp[j]=max(dp[j],dp[j-time]+value)
print(dp[-1])
```





### M20744: 土豪购物

dp, http://cs101.openjudge.cn/practice/20744/

思路：双DP



代码：

```python
lst=list(map(int,input().split(',')))
n=len(lst)
dp1=[0 for _ in range(n)]
dp2=[0 for _ in range(n)]
dp1[0]=lst[0]
ans=dp1[0]
dp2[-1]=lst[-1]
for i in range(1,n):
    dp1[i]=max(dp1[i-1]+lst[i],lst[i])
    dp2[-i-1]=max(dp2[-i]+lst[-i-1],lst[-i-1])
    ans=max(dp1[i],ans)
for i in range(1,n-1):
    ans=max(ans,dp1[i-1]+dp2[i+1])
print(ans)
```





### 04102:宠物小精灵之收服

 http://cs101.openjudge.cn/practice/04102/

```
limit,blood,num=map(int,input().split())
INF=float('inf')
dp=[[INF for _ in range(blood+1)] for _ in range(num+1)]#值表示损失的精灵球，列坐标表示损失血量
dp[0][0]=0
for i in range(1,num+1):
    cost,harm=map(int,input().split())
    for j in range(i,-1,-1):
        for k in range(blood,harm-1,-1):
            if 0<=dp[j-1][k-harm]<=limit-cost:
                dp[j][k]=min(dp[j-1][k-harm]+cost,dp[j][k])
for i in range(num,-1,-1):
    for j in range(blood+1):
        if dp[i][j]!=INF:
            print(i,blood-j)
            exit()
```



### 01664：放苹果

 http://cs101.openjudge.cn/practice/01664/

```python
    dp=[[0 for _ in range(n+1)]for _ in range(m+1)]#值表示放法，行数表示苹果数，列数表示盘子数
    for i in range(1,n+1):
        dp[0][i]=1#零个苹果，放法均为1
    for i in range(1,m+1):
        if i>=n:#如果苹果数多于盘子数
            for j in range(1,n+1):
                dp[i][j]+=dp[i][j-1]+dp[i-j][j]#放n-1个盘子的情况加上放n个盘子的情况，后者等价于先在这n个盘子上各放一个，再将i-j个苹果放在n个盘子上
        else:
            for j in range(1,i+1):
                dp[i][j]+=dp[i][j-1]+dp[i-j][j]
            for j in range(i+1,n+1):#盘子数多于苹果数，直接等于盘子数与苹果数相同
                dp[i][j]+=dp[i][j-1]
    return dp
```

## 

### 25573：红蓝玫瑰

 http://cs101.openjudge.cn/practice/25573/

```python
r=list(input())
n=len(r)
R=[0]*n#全变红
B=[0]*n#全变蓝
if r[0]=="R":
    R[0]=0;B[0]=1
else:
    R[0]=1;B[0]=0
for i in range(n-1):
    if r[i+1]=="R":
        R[i+1]=R[i]
        B[i+1]=min(R[i],B[i])+1
    else:
        R[i+1]=min(R[i],B[i])+1
        B[i+1]=B[i]
print(R[-1])
```

