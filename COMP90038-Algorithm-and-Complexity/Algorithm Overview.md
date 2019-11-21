- COMP90038 Algorithm and Complexity Review Notes ---
- Author：Chongzheng Zhao（https://github.com/ChongzhengZhao/）
- Algorithm Analysis
	- Data Structure(Array, Link List, Stack, Queue, Priority Queue, Binary Tree, Hashing)
		- Array
			[Search Pseudocode] - Recursive
			function find(A,x,lo,hi)   lo->0 hi->n-1
				if lo > hi
					return -1
				else if lo = A[lo]
					return lo
				else
					return find(A,x,lo+1,hi)
		- Link List
			Node and Pointer(The pointer in last node is null)
			[Function]
				Value - ListName.val
				Next Value - ListName.next
			[Search Pseudocode] - Recursive
			function find(p,x)    p->head
				if p = null then
					return p
				else if p.val = x
					return p
				else
					return find(p.next,x)
		- Stack
			[Property] Last In First Out
			[Function]
				Initial - initilise st as an empty stack
				Add - st.push(element)
				Out - st.pop()
				Null judgement - if st is empty
		- Queue
			[Property] First In First Out
			[Function]
				Initial - init(queue)
				Add - inject(queue,element)
				Out - eject(queue)
				Null judgement - if queue is empty
				Head - queue.head()
		- Binary Tree
			[Property]
				Empty Tree Height = -1
			[Traveral]
				Pre-order
				In-order
				Post-order
		- Binary Search Tree [Based on Binary Tree]
			[Property]
				Left Child is LESS THAN Parents;
				Right Child is LARGER (OR EQUAL) TO Parents;
			[Count Quantity of Diff elements]
				B(n+1) = B(n) * B(0) + B(n-1) * (1) ... + B(0) * B(n)
		- (AVL)Balanced Binary Search Tree [Based on Binary Search Tree]
			[Property]
				Left Child is LESS THAN Parents;
				Right Child is LARGER (OR EQUAL) TO Parents;
				[!]The |difference between two children| must be <= 1
			[Balanced Rule] (>>Dueape<<P76Figure)
				L-Rotation
				R-Rotation
				LR-Rotation
				RL-Rotation
		- (2-3Tree)Balanced Binary Search Tree [Based on Binary Search Tree]
		
		- Priority Queue(max heap & min heap)
			- [Time Complexity]
				Add element - O(logn)
				Delete the heighest priority element - O(logn)
				Add elements to empty heap - O(nlogn)
				Create a complete binary tree first and then construct from bottom up - O(n)
		- Hashing (Use Space to Reduce Time)
			- [Handle Collision]
				- Separate Chain : Use link list
					Load Factor(alpha) = n/m ; n:the number of items stored, m: the table size
					Number of probes in successful search: (1+alpha)/2
		  			Number of probes in unsuccessful search: alpha
				- Open-addressing methods
					- Linear probing : Store in the next index
					Number of probes in successful search: 0.5+1/(2(1-alpha))
					Number of probes in unsuccessful search: 0.5+1/(2(1-alpha)^2)
					#PS 在Linear Probe中查找一个已经被移过位置的key是unsuccessful probing
					- Double hashing : Use another hashing function s(k) to add to the original index[h(k)+s(k)]. If collision again, add 2s(k)......
		- Graphs
			[Property] 
				- Vertex[V], Edges[E], Weight --->>> G<V,E>
				- Edge(u,v) means the edge between node u and node v
				- two nodes is connected, then we call v and u are ADJACENT or NEIGHBOURS
				- Degree : NUMBER of EDGES connected on a node
					- In-degree : NUMBER of EDGES GOING TO v
					- Out-degree : NUMBER of EDGES LEAVING FROM v
			[Graph Representation] (>>Dueape<<P34Figure)
				- Adjacency Matrix
					- 0 means not connected, 1 means they are connected
					- for undirrected graph, the AM is Symmetric(对称)
				- Adjacency List
					- e.g (a -> b -> c -> d)
	- Calculation of Time Complexity[t(n)]
		- Normal Algorithm
			- How to find the basic operation:
				the innermost
			- Expression of Time Complexity
				- O : O(g(n)) means TC is slower than g(n)
				- Omega : O(g(n)) means TC is faster than g(n)
				- Theta : O(g(n)) means TC is same than g(n)
			- Calculation Step
				- Basic Operation Efforts * Numbers of execution
				- lim n->Infinity / 求导数
					- Concept
						Ignore constant factors
						Ignore small input sizes
						Think n bigger!!!
						(!)1 ≺loglogn ≺logn ≺n^ε ≺n^c ≺n^logn ≺c^n ≺n^n 
							where 0 < ε < 1 < c
		- Recursive Algorithm
			- TC consist of TWO PARTS : ENDING POINT EFFORT + RECURSIVE EFFORT
			- EXAMPLE (>>Dueape<<P21)
			- Use Telescoping to Calculate the RECURSIVE EFFORT (>>Dueape<<P21)
		- Master Theorem
			- (!)T(n) = aT(n/b) + f(n)  f(n) Belongs To Theta(n^d)
				>> T(n) = Theta(n^d) if a < b^d
				>> T(n) = Theta(logn * n^d) if a = b^d
				>> T(n) = Theta(n^logb(a)) if a > b^d
				# 记住是Theta，记住内容
- Sorting Algorithm
	- SelectSort (Brute Force)
		[ABS] Swap the index one and the Selected the smallest one
		[Pseudocode]
			function SelectSort(A[~],n) n<-length
				for i <- 0 to n-2 do
					min <- i
					for j <- i+1 to n-1 do
						if A[j] < A[min] then    //Find the smallest one!
							min <- j
					t <- A[i]     //Swap A[i] and A[min]
					A[i] <- A[min]
					A[min] <- t
		[Time Complexity]
			Basic Operation(A[j] < A[min]), its efforts(1)
			Worst recursive times(n^2)
			So worst TC belongs to O(n^2)
		[Property]
			In-Place? Y
			Stable? N
			Input-insensitive? Y(No sensitive)
	- InsertionSort (Decrease and Concur)
		Take the index one, and downward to left to find a place to insert.(swap the elements inside the scope if they are not the target to be inserted in)
		[Pseudocode]
			function insertionsort(A[~],n)
				for i <- 1 to n-1 do
					v <- A[i]   //take the index one
					j <- i - 1  //make a index of comparison target
					while j >= 0 and v < A[j] do // skip those non-target
						A[j+1] <- A[j]	//swap the non-target to right to make a place
						j <- j-1 //downward then
					A[j+1] <- v //insert the target to the place(already empty)
		[Improvement]
			when the minimum element is known, we can put it in the first element, so the j >= 0 statement can be thowrn, just keep v < A[j] is enough.
		[Time Complexity]
			Basic Operation(v < A[j]), its efforts(2)
			Worst(Reverse order) recursive times(n)
			So worst TC belongs to O(2n*n) ->> O(n^2)
			Best recursive times(0)
			So best TC belongs to O(n)
		[Property]
			In-Place? Y!
			Stable? Y
			Input-insensitive? N(sensitive)
	- ShellSort (Based on InsertionSort)(Decrease and Concur)
		Based on InsertionSort, it first evenly grouped the elements into K sub-group, and in each group execute the insertionsort, and finally use the insertionsort to finalise the overall result. ##Left element has priority!See code!
		[Pseudocode] NOT EXAMINABLE

		[Time Complexity]
		Worst case TC belongs to O(nSQRT(n))
		[Property]
			In-Place? Y
			Stable? N
			Input-insensitive? N(sensitive)
	- MergeSort (Divide and concur) - inluding divide and combine
		First divide into piece of size 1, and then do the merge function(Visually, it is from the left most two pair(initial size is 1) and sorting). 
		[Pseudocode] - Divide
			function mergesort(A[.],n)
				if n > 1 then		//State the stopping condition
					for i <- 0 to BOTTOM(n/2 -1) do 	// Divide the left part to B
						B[i] <- A[i]
					for i <- 0 to UPPER(n/2 - 1) do 	// Divide the right part to C
						C[i] <- A[BOTTOM(n/2) + i]
					mergesort(B[.],BOTTOM(n/2))
					mergesort(C[.],UPPER(n/2))
					merge(B[.],BOTTOM(n/2),C[.],UPPER(n/2),A[.])	//Combine
		[Pseudocode] - Combine
			function merge(B[.],p,C[.],q,A[.])
				i <- 0, j <- 0, k <- 0		//init three indexes
				while i < p and j < q do     //avoid out of bounds
				//copy the minimal element to A
				//see <= , so left element has priority to copy to A when B&C same
					if B[i] <= C[j]		
						A[k] <- B[i]
						i <- i + 1
					else
						A[k] <- C[j]
						j <- j + 1
					k <- k + 1
				if i = p then
					copy C[j]...C[q-1] to A[k]...A[p+q-1]
				else
					copy B[i]...B[p-1] to A[k]...A[p+q-1]
		[Time Complexity]
			Divide costs logn, Merge costs n
			Total TC belongs to Theta(nlogn)
		[Property]
			In-Place? N 
			Stable? Y (because left element always has priority)
			Input-insensitive? Y(insensitive)
	- QuickSort (Divide and concur) - inluding divide and combine - most efficient one
		[Pseudocode] - Main
			function quicksort(A[.],lo,hi)
				if lo < hi then
					s <- partition(A,lo,hi)  // Remember to assigned the return to s
					quicksort(A,lo,s-1)
					quicksort(A,s+1,hi)  // Remember it is s+1
		[Pseudocode] - Partition
			function partition(A[.],lo,hi)
				p <- A[lo]
				i <- lo
				j <- hi
				repeat
					while i < hi and A[i] <= p do i <- i + 1
					while j >= lo and A[j] > p do j <- j - 1
					swap(A[i],A[j])
				until i >= j
				swap(A[i],A[j])    // undo last swap
				swap(A[lo],A[j]) 	// bring pivot to j position
				return j
		[Improvement 1] >>Dueape<<P61
		When analysing quicksort in the lecture, we noticed that an already sorted array is a worst-case input. Is that still true if we use median-of three pivot selection?
		Answer: This is no longer a worst case; in fact it becomes a best case! In this case the median-of-three is in fact the array’s median. Hence each of the two recursive calls will be given an array of length at most n/2, where n is the length of the whole array. And the arrays passed to the recursive calls are again already-sorted, so the phenomenon is invariant throughout the calls.
		[Improvement 2]
			>>Dueape<<P62
		[Time Complexity]
			Worst Case O(n^2)
			Best Case O(nlogn)
		[Property]
			In-Place? Y
			Stable? N!
			Input-insensitive? N(sensitive)
	- HeapSort (Transform and Concur)
		和二叉搜索树不一样！当是最大堆的时候，某节点的两个孩子都是比爸爸小的！！
		Heapsort = 构建最大堆 + 删除最高优先级元素
		[Time Complexity]
			Insert elements to a heap costs logn + Make it Heighest Heap cost n + deltete the highest priority element cost logn
			TC OVERALL O(nlogn)
		[Step]
			1.插入后的最大堆平衡 ：右 -> 左 -> 右（最下面的孩子->父母（若有变）->孩子重新回去验证）
			2.删除最高优先级元素：第一个元素放在最末尾，然后按照插入的步骤重新平衡验证（次最大的接着放在倒数第二个，以此类推）
		[Property]
			In-Place? Y
			Stable? N
			Input-insensitive? Y!(insensitive)
	- Sort by Couting (Use Space to Reduce Time)
		[Must Requirement] the length N >> number scope! It is not efficient to use when k > nlogn!
		[Time Complexity and steps]
			OVERALL COMPLEXITY IS O(n+k) ->> O(n)
				- construct a array B(length k+1), initial all value to 0.[cost O(k+1)]
				- from left to right to traverse, and count each element appear times to record in B.[cost O(n)]
				- In B, accumulation from left to right.[cost O(k)]
				- Construct a array C(length n), left its defalt. [cost O(1)]
				- Regard the value of B as index in C, add the value to C.[cost O(n)]
				- Overall, cost O(2k+2n+2) ->> O(n+k) ->> O(n) [Linear Complexity]
	C- Comparison of Sorting algorithms
						In-Place? | Stable? | Input-Insensitive? | Time Complexity
	BF	SelectSort  |		Y	  |	   N    |     	  Y			 |     O(n^2)
	DeC	InsertiSort |		Y	  |	   Y!   |         N!         |Be-O(n) Wor-O(n^2)
	DeC	ShellSort	|		Y     |    N    |         N!         |Wor-O(nSQRT(n))
	DiC	mergesort   | 		N     |    Y    |         Y          |	   O(nlogn)
	DiC	quicksort   |		Y     |	   N    |		  N          |Be-O(nlogn)Wor-O(n^2)
	TrC	HeapSort	| 		Y	  |    N    |         Y			 |	   O(nlogn)
	
	#2017Sample# Stable means the order of input elements is unchanged except where change is required to satisfy the requirements. A stable sort applied to a sequence of equal elements will not change their order.

	#2017Sample# In-place means that the input and output occupy the same memory storage space. There is no copying of input to output, and the input ceases to exist unless you have made a backup copy. This is a property that often requires an imperative language to express, because pure functional languages do no have a notion of storage space or overwriting data.
- Searching Algorithm 
	- Binary Search (Decrease and Concur)
		[Requirement]
			Given a PRESORTED array can apply BinSearch
			Complexity will increase if given LINK LIST(because cannot be indexed)
		[Psedocode] - recursive
			function BinSearch(A[.],lo,hi,key)
				if lo > hi then
					return -1
				mid <- BOTTOM((lo+hi)/2)
				if A[mid] = key then
					return mid
				else
					if A[mid] > key then
						return BinSearch(A,lo,mid-1,key)
					else
						return BinSearch(A,mid+1,hi,key)
		[Time Complexity]
			C(n) belongs to O(logn)
	- QuickSelect(Find the K Smallest Element) (Decrease and Concur)(Tag:find median)
		[!] No need to pre-sorted, suit for just search in unsorted array(in sorted array is the best case)
		[Psedocode] - Main
			function QuickSelect(A[.],lo,hi,k)
			s <- LomutoPartition(A,lo,hi)	//know the first element is WHICH LARGEST
			if s - lo = k - 1 then    //k-1 means the index of k smallest elements
			//this means the previos "first element" is coincidently the k smallest ele
				return A[s]
			else
				if s - lo > k - 1 then  //this means the k smallest is located before s
					QuickSelect(A,lo,s-1,k)
				else     // else, it is located after s
					QuickSelect(A,s+1,hi,(k-1)-(s-lo))
		[Psedocode] - LomutoPartition
			//return the first element is the k largest and make every element which small than k is located at the left side of k, while the larger element than k is located at the right hand side of k
			function LomutoPartition(A[.],lo,hi)
				p <- A[lo]			//first element, set as pivot
				s <- lo 		//the index
				for i <- lo + 1 to hi do //begin from the second element
					if A[i] < p then	//if [i] element is smaller than the first eles
						s <- s + 1		// first add the index
						swap(A[s],A[i]) //then swap
				swap(A[s],A[lo])  //make first element to locate at the TARGET location
				return s  //make sure return current index of previous "first" element
		[Time Complexity]
			Best case Complexity belongs to O(n)
			Worst case Complexity belongs to O(n^2)
	- Interpolation Search (Decrease and Concur BY variable)(Based On BinSearch)
		[Requirement, Suit for]
			PRE-SORTED Array; 
			Suit for BIG LENGTH ARRAY but elements are EVENLY DISTRIBUTED
		[Improvement on BinSearch]
			m <- (lo+hi)/2  ----->>>>>>  m <- lo + (k-A[lo])/(A[hi]-A[lo])(hi-lo)
			把硬核的m直接算均数 变成 k在数组中的#相对比例#位置
		[Time Complexity]
			C(n) belongs to O(loglogn)  >- Pretty small! -<
- Graph Algorithm
	- Deepth-First search (Brute Force) [Based on STACK]
		[Property]
			[!]Use DFS can make an judgement on WHETHER the graph is a circle(Instead of marking visited node with consecutive integers, we can mark them with a number that identifies their connected component. More specifically, replace the variable count with a variable component. In dfs remove the line that increments count. As before, initialise each node with a mark of 0 (for “unvisited”).[[JUST MOVE COUNT<-COUNT+1 to plact at AFTER DFSEXPOLRE(v)]])
			P.S can use a individual array to STORE which COUNT belongs to which NODE
		[Pseudocode]
			function DFS(<V,E>)
				mark each node in V with 0   //initial all node count to 0
				count <- 0    // initial a count
				for each v in V do    //traverse all the nodes
					if v is marked with 0 then    //if v not visited then
						DFSExplore(v)
											// if count++ place here can achieve judgement function(SEE UPPER DETAILS`)
			function DFSExplore(v)
				count <- count + 1   //in order to give a unique identification
				mark v with count   //mark it!
				for each Eege(v,w) do  //then continue to impact v's conjacency
					if w is marked with 0 then //if a neighbour not been visited then
						DFSExplorer(w)
		[Time Complexity]
			If use Conjacancy Matric, C(n) belongs to Theta(|V|^2)
			If use Conjacancy List, C(n) belongs to Theta(|V|+|E|)
		[DFS Forest] (>>Deuape<<P39Figure)
	- Breath-first search (Brute Force) [Based on QUEUE]
		[Property]
		[!] Use BFS can find the SMALLEST length between two nodes
		P.S can use a individual array to STORE which COUNT belongs to which NODE
		[Pseudocode]
		function BFS(<V,E>)
			mark each node in V with 0		//initial all node count to 0
			count <- 0 		// initial a count
			init(queue)		// BFS based on queue, init it
			for each v in V do  // traver all nodes
				if v is marked with 0 then   // if v is not visited
					count <- count + 1 	//in order to give a unique identification
					mark v with count    //Mark it!
					inject(queue,v) 	//UNIQUE! INJECT IT TO QUEUE
					while queue is not empty do //UNIQUE! Check queue empty
						u <- eject(queue)	//UNIQUE! inject one node instead of v
						for each Edge(u,w) do   //UNIQUE! it is u! not v!
							if w is marked with 0 then
											//if a neighbour not been visited then
								count <- count + 1 //UNIQUE! do count + 1 again
								mark w with count //UNIQUE! do mark in a function again
								inject(queue,w)	
											//UNIQUE! inject it instead of recursive!
		[Time Complexity] Same as DFS
		[BFS Forest] (>>Deuape<<P41Figure)
	- Topological Sorting (Decreae and Concur) 
		[Property]
		[!] ONLY FOR DIRRECTED UNCIRCAL GRAPH（有向无环图）
		[!] MUST BE OBEY for edge(v1,v2), v2 MUST BE LOCATED BEFORE v2
		Results are multiple, not unique.
		[Steps]
		解法1:使用#DFS，顶点的#退出顺序的#反序则是拓扑排序顺序
		解法2:移除目前的#只有向外指出edge的node，node被移除的顺序就是拓扑排序顺序
	- Warshall's Algorithm(求有向图的传递闭包) （Dynamic Programming)
		[Signal]Is there a path from node i to node j using nodes[1...k] as "Step Stones?"?
		[Purpose] 有向图，找两个Node可否直接（间接）相连
		[Pseudocode]
			function Warshall(A[.,.],n)
				R[.,.,0] <- A
				for k <- 1 to n do
					for i <- 1 to n do
						for j <- 1 to n do
							R[i,j,k] <- R[i,j,k-1] or (R[i,k,k-1] and R[k,j,k-1])
				return R[.,.,n]
		[Time Complexity]
			Theta(n^3)
		[Graph Example] 
			>>Dueape<<P92Figure
			从左上角开始向右下角行列扫描，1,1 <- 1 ; 0,1 <- 0 ; 0,0 <- 0
	- Floyd's Algorithm(求全源最短路径)  (Dynamic Programming)
		[Singal]What is the shortest path from node i to node j using nodes[1...k] as "Step Stones"?
		[Purpose] （无向图）找两个Node之间最短的距离
		[Pseudocode]
			function Floyd(W[.,.],n)
				D <- W
				for k <- 1 to n do
					for i <- 1 to n do
						for j<-1 to n do
							D[i,j] <- min(D[i,j], D[i,k]+D[k,j])
				return D
		[Time Complexity]
			Theta(n^3)
		[Graph Example] 
			>>Dueape<<P102Figure
			从左上角开始向右下角行列扫描，对角线对应赋值最小值
	- Prim's Algorithm（最小生成树算法） (Greedy Algorithm)
		[Purpose]无向图，找最小路径生成树（和Dijkstra相似，但是每次寻遍的时候Node不与之前的Node叠加，因为不是找node到node[Pseudocode]
		[Time Complexity] 
			(|V|-1+|E|)O(log|V|)
			(!)连通图中|E| > |V|-1 , so TC is O(|E|log|V|)
		[Graph Example]
			>>Dueape<<P102Figure110
	- Dijkstra's Algorithm (求单源最短路径) (Greedy Algorithm)
		[Purpose]有向图，找某node到任何地方的最短路径
		!!!!![Pseudocode] NO IDEA WHETHER IT IS EXAMNABLE
		[Time Complexity] The same as Prim
		[Graph Example]
			>>Dueape<<P102Figure119
- String Matching Algorithm
	- Brute Force String Maching (Brute Force)
		[Pseudocode]
			// S <- String, t <- Taget String, m & n is length of them
			function BruteStringMatching(s[.],m,t[.],n)
				for i <- 0 to m - n do    //only to m - n!
					j <- 0	
					while j < n and t[j] = s[i+j]  //avoid out of bounds and judge
						j <- j + 1
					if j = n     
								//if the j = n means the num of matching = number of target string
						return i
				return -1
		[Time Complexity]
			C(n) belongs to O(mn)
	- Hospool String Matching (Use Space to Reduce Time)
		[Mechanism]
			1.Set up #Shift table, containing all the unque number.
			2.index the original string from left to right a DESCENT NUMBER to ZERO eventually.
			3.from left to right UPDATE the index in SHIFT TABLE, but NOT VALID with INDEX ZERO
		[Pseudocode]
			NO IDEA WHETHER IT IS EXAMNABLE
		[Time Complexity]
			Worst case complexity is O(mn)
- Closed Pair Algorithm (Brute Force)
	寻找距离最短的两个坐标！
	[Psedudocode]
		function ClosedPair(A[.,.])
			min <- Infinity		//init a min assigned to infinity
			for i <- 0 to n-2 do 	
								// to n-2 because the last index will one smaller than j
				for j <- i + 1 to n-1 do // begin from i+1 because it starts after i
					d <- Sqrt((xi-xj)^2-(yi-yj)^2)
					if d < min then
						min <- d
						p <- i
						q <- j
			return p,q
	[Time Compelxity]
		Basic Operation is cal of d and comparision of d : 2
		Times : n^2
		TC: O(n^2)
- Dynamic Programming
	- Coin Problem >>Slides<<P85
		[Recursive Function]
			S(n) = max(S(n-1), S(n-2) + cn)
			S(0) = 0
			S(1) = 1
		[Steps]
			State  0 1 2 3 4 5 6 ........
			Coin   x x x x x x x ........
			take   y y y y y y y ........
			notake z z z z z z z ........
			   	[Every time choose the max from take and notake as this index status]
	- Packbag Problem >>Slides<<P86
		[Recursive Function]
			Base Case : K(i,w) = 0 if i = 0 or w = 0 
			K(i,w) = max(K(i-1,w),K(i-1,w-wi)+vi) if w >= wi
			K(i,w) = K(i-1,w) if w < wi
		[Steps]
				j   0	1	2	3	4	5	6	7	8	9 ... (to the bag capacity W)
		v | w | i
				0	0	0	0	0	0	0	0	0	0	0
		x   x   1 	y 	y 	y 	y 	y 	y 	y 	y 	y 	y  [上一行-w位置的值+这行的v]
				2				.							[每一次与上行相比取最大值]
				3				.
				4				.

	- Warshall's Algorithm(求有向图的传递闭包) [See Graph Algorithm]
	- Floyd's Algorithm(求全源最短路径)	[See Graph Algorithm]
- Greedy Algorithm （Greedy cannot always give the optimal solution)
	- Prim's Algorithm（最小生成树算法）[See Graph Algorithm]
	- Dijkstra's Algorithm (求单源最短路径) [See Graph Algorithm]




======Extra
- sequential search in list and array:
	we can use binary search if it is an array(apply the index to find)
	we can stop immediately as soon as we find that element.(but we cannot use index)
- Limitation of Binary Search Tree
	 当搜索的任务与值的大小无关的时候（比如查找二叉树中第一个能被5整除的数），二叉搜索树就没有什么用了。
	 如果输入的顺序是sorted输入，那么height将会非常的高（和数组没什么区别了）
- 等比数列、等差数列求和公式
	- 等差数列
		S(n)= n(a1 + an)/2
	- 等比数列
		S(n)= a1(1-q^n)/(1-q)





