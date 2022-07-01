# Python3 program to find largest rectangle
# with all 1s in a binary matrix

# Finds the maximum area under the
# histogram represented
# by histogram. See below article for details.


class Solution():
    def maxHist(self, row):
        result = []
        # Top of stack
        top_val = 0
        top_val_0 = 0 

        # Initialize max area in current
        max_area_1 = 0
        area_1 = 0  # Initialize area with current top

        i = 0
        while (i < len(row)):
            print("start", i )
            print('len result', len(result))
            # If this bar is higher than the
            # bar on top stack, push it to stack
            if (len(result) == 0) or (row[result[-1]] <= row[i]):
                result.append(i)
                result_0 = result[::-1]
                i += 1
                print('result', result)
                # print('len result', len(result))
            else:

                # print('enter else')
                top_val = row[result.pop()]
                # print('len result_2', len(result))
                print(i, "top_val", top_val)
                area_1 = top_val * i
                # print("area", area)

                if (len(result)):
                    area_1 = top_val * (i - result[-1] - 1)
                    print("area", area_1)
                max_area_1 = max(area_1, max_area_1)


        print("END")
        while (len(result)):
            top_val = row[result.pop()]
            print('result final', result)
            area_1 = top_val * i
            if (len(result)):
                area_1 = top_val * (i - result[-1] - 1)

            max_area_1 = max(area_1, max_area_1)

        return max_area_1

    # Returns area of the largest rectangle
    # with all 1s in A
    def maxRectangle(self, A):
        # A_flip = []
        rows, cols = (len(A), len(A[0]))
        A_flip = [[0]*cols]*rows
        # print(A_flip)
        # print(A)
        for i in range(len(A)) :
            A_flip[i] = [1- x for x in A[i]]

        print('A_flip', A_flip)
        # A_flip = [[1, 0, 0, 1],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 1, 1]]

        result_1 = self.maxHist(A[0])
        result_0 = self.maxHist(A_flip[0])

        # iterate over row to find maximum rectangular
        # area considering each row as histogram
        
        for i in range(1, len(A)):

            for j in range(len(A[i])):

                # if A[i][j] is 1 then add A[i -1][j]
                if (A[i][j]):
                    A[i][j] += A[i - 1][j]

            for j in range(len(A_flip[i])):

                # if A[i][j] is 1 then add A[i -1][j]
                if (A_flip[i][j]):
                    A_flip[i][j] += A_flip[i - 1][j]

            # Update result 

            result_1 = max(result_1, self.maxHist(A[i]))
            result_0 = max(result_0, self.maxHist(A_flip[i]))

        # return result
        return max(result_1, result_0)


# Driver Code
if __name__ == '__main__':
    # A = [[0, 1, 1, 0],
    #      [1, 1, 1, 1],
    #      [1, 1, 1, 1],
    #      [1, 1, 0, 0]]

    A = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 1, 0, 1, 1],
         [0, 0, 0, 1, 1],
         [1, 1, 0, 1, 1]]

    ans = Solution()

    print("Area of maximum rectangle is",
          ans.maxRectangle(A))
/home/huyen/Desktop/Ex_max_rec.py