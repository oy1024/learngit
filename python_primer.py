"""
    用python实现对特定范围内的数据的素数进行筛选
    角度：面向对象的方法
"""



#从三开始建立一个非偶数队列
def build_number():
    i=1
    while True:
        i = i + 2
        yield i

#选出队列中要删除的数据
def not_able(i):
         #采用匿名函数，返回计算的数据
    return lambda x:x % i > 0

#使用filter()函数求素数
def primes():
    yield 2
    it = build_number()
    while True:
        i = next(it)
        yield i
        it = filter(not_able(i),it)

def main():
#输出1000以内的素数
    for i in primes():
        if i < 10000:
            print(i)
        else:
            break

if __name__ =='__mian__':
    main()
