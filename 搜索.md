# 搜索

### 04129: 变换的迷宫

bfs, http://cs101.openjudge.cn/practice/04129/

思路：三维visited



代码：

```python
from collections import deque


def bfs(rows, cols, k, maze, start_x, start_y):
    dx=[0,0,1,-1]
    dy=[1,-1,0,0]
    visited=set()
    visited.add((0,start_x,start_y))
    stack=deque()
    stack.append((0,start_x,start_y))
    while stack:
        num=len(stack)
        while num:
            num-=1
            cnt,front_x,front_y=stack.popleft()
            if maze[front_x][front_y]== 'E':
                return cnt
            temp=(cnt+1)%k
            for o in range(4):
                nx,ny= front_x + dx[o], front_y + dy[o]
                if 0<=nx<rows and 0<=ny<cols and (maze[nx][ny] != '#' or temp == 0) and(temp, nx, ny) not in visited:
                    stack.append((cnt+1,nx,ny))
                    visited.add((temp,nx,ny))
    return 'Oop!'

t=int(input())
ans=[]
for _ in range(t):
    R,C,K=map(int,input().split())
    matrix=[list(str(input())) for _ in range(R)]
    for i in range(R):
        for j in range(C):
            if matrix[i][j]=='S':
                ans.append(bfs(R,C,K,matrix,i,j))
                break
print(*ans,sep='\n')
```

### 18160: 最大连通域面积

dfs similar, http://cs101.openjudge.cn/practice/18160

思路：dfs常见模板



代码：

```
dr=[-1,-1,-1,0,0,1,1,1]
dc=[-1,0,1,-1,1,-1,0,1]
nums=0
def dfs(start_r,start_c,rows,cols,matrix):
    global nums
    for k in range(8):
        nr=dr[k]+start_r
        nc=dc[k]+start_c
        if 0<=nr<rows and 0<=nc<cols and matrix[nr][nc]=='W':
            matrix[nr][nc]='.'
            nums+=1
            dfs(nr,nc,rows,cols,matrix)

t=int(input())
ans=[]
for _ in range(t):
    row,col=map(int,input().split())
    area=[list(input())for _ in range(row)]
    res=0
    for i in range(row):
        for j in range(col):
            if area[i][j]=='W':
                nums=1
                area[i][j]='.'
                dfs(i,j,row,col,area)
                res=max(res,nums)
    ans.append(res)
print(*ans,sep='\n')
```





代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20241120102917736](C:\Users\谭宇睿\AppData\Roaming\Typora\typora-user-images\image-20241120102917736.png)



### 19930: 寻宝

bfs, http://cs101.openjudge.cn/practice/19930

思路：bfs常见模板



代码：

```python
from collections import deque


def bfs(start_r,start_c,rows,cols,matrix,visited):
    x=[-1,0,0,1]
    y=[0,1,-1,0]
    step=0
    q=deque()
    q.append((start_r,start_c))
    while q:
        cnt=len(q)
        while cnt>0:
            front=q.popleft()
            r,c=front
            if matrix[r][c]==1:
                return step
            cnt-=1
            for i in range(4):
                nr=r+x[i]
                nc=c+y[i]
                if 0<=nr<rows and 0<=nc<cols and matrix[nr][nc]!=2 and not visited[nr][nc]:
                    q.append((nr,nc))
                    visited[nr][nc]=True
        step+=1
    return 'NO'

m,n=map(int,input().split())
area=[list(map(int,input().split()))for _ in range(m)]
inq=[[False for _ in range(n)]for _ in range(m)]
inq[0][0]=True
print(bfs(0,0,m,n,area,inq))
```



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123

思路：注意绊脚



代码：

```python
dr=[-2,-2,-1,-1,1,1,2,2]
dc=[-1,1,-2,2,-2,2,-1,1]
res=0
def dfs(rows,cols,start_r,start_c,visited,step):
    global res
    if step==rows*cols:
        res+=1
    for k in range(8):
        nr=dr[k]+start_r
        nc=dc[k]+start_c
        if 0<=nr<rows and 0<=nc<cols and not visited[nr][nc]:
            visited[nr][nc]=True
            dfs(rows,cols,nr,nc,visited,step+1)
            visited[nr][nc]=False

t=int(input())
ans=[]
for _ in range(t):
    row,col,r,c=map(int,input().split())
    inq=[[False for _ in range(col)]for _ in range(row)]
    inq[r][c]=True
    dfs(row,col,r,c,inq,1)
    ans.append(res)
    res=0
print(*ans,sep='\n')
```



### sy316: 矩阵最大权值路径

dfs, https://sunnywhy.com/sfbj/8/1/316

思路：

注意负数

代码：

```python
max_sum=-float('inf')
ans=[]
dx=[-1,0,0,1]
dy=[0,-1,1,0]
n,m=map(int,input().split())
dungeon=[list(map(int,input().split()))for _ in range(n)]
inq=[[False for _ in range(m)]for _ in range(n)]
inq[0][0]=True
ans.append((0,0))
def dfs(x,y,matrix,rows,cols,steps,visited,temp):
    global max_sum
    if x==rows-1 and y==cols-1:
        if max_sum<steps:
            max_sum=steps
            ans[:]=temp
        return
    for k in range(4):
        nx=dx[k]+x
        ny=dy[k]+y
        if 0<=nx<rows and 0<=ny<cols and not visited[nx][ny]:
            visited[nx][ny]=True
            temp.append((nx,ny))
            dfs(nx,ny,matrix,rows,cols,steps+matrix[nx][ny],visited,temp)
            visited[nx][ny]=False
            temp.pop()
dfs(0,0,dungeon,n,m,dungeon[0][0],inq,[(0,0)])
for i in ans:
    a,b=i
    print(a+1,b+1)
```

### 25572: 螃蟹采蘑菇

bfs, dfs, http://cs101.openjudge.cn/practice/25572/

思路：bfs,先遍历矩阵找到两个起点，再找出两者关系，之后只对其中一个进行重点关注



代码：

```python
from collections import deque

n=int(input())
maze=[list(map(int,input().split())) for _ in range(n)]
start=[]
for i in range(n):
    for j in range(n):
        if maze[i][j]==5:
            start.append((i,j))
delta_x=start[1][0]-start[0][0]
delta_y=start[1][1]-start[0][1]
visited=set()
visited.add((start[0][0],start[0][1]))

def is_valid(r,c):
    if 0<=r<n and 0<=c<n and (r,c) not in visited and maze[r][c]!=1 and 0<=r+delta_x<n and 0<=c+delta_y<n and maze[r+delta_x][c+delta_y]!=1:
        return 1
    else:
        return 0

dx=[0,0,1,-1]
dy=[1,-1,0,0]
stack=deque()
stack.append((start[0][0],start[0][1]))
while stack:
    front_x,front_y=stack.popleft()
    if maze[front_x][front_y]==9 or maze[front_x+delta_x][front_y+delta_y]==9:
        print('yes')
        exit()
    for i in range(4):
        nx,ny=front_x+dx[i],front_y+dy[i]
        if is_valid(nx,ny):
            visited.add((nx,ny))
            stack.append((nx,ny))
print('no')
```

### sy358: 受到祝福的平方

dfs, dp, https://sunnywhy.com/sfbj/8/3/539

思路：



代码：

```python
import math
def is_sqrt(n):
    if math.sqrt(int(n)).is_integer() and int(n)>0:
        return 1
    else:
        return 0
ans='No'
S=str(input())
t=len(S)
def dfs(s,steps):
    global ans
    if steps==t:
        if len(s)==0:
            ans='Yes'
        return
    i=-1
    while i+len(s)>=0:
        if is_sqrt(s[i:]):
            dfs(s[:i],steps-i)
        i-=1
dfs(S,0)
print(ans)
```

