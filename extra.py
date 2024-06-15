class Solution:
    def calculate(self, s: str) -> int:

        def getNext(s):

            while len(s) > 0 and s[0] == ' ':
                s = s[1:]

            if len(s) > 0 and s[0] in ['+', '-', '(', ')']:
                symbol = s[0]
                s = s[1:]
                return symbol, s

            num = ""
            while len(s) > 0 and ord(s[0]) >= ord('0') and ord(s[0]) <= ord('9'):
                num += s[0]
                s = s[1:]

            if num == "":
                return 'end', ""

            return num, s

        def evaluate(exp):

            print("exp")

            while len(exp) > 1:
                print(exp)
                val1 = exp.pop(0)
                op = exp.pop(0)
                val2 = exp.pop(0)

                if op == '+':
                    res = int(val1) + int(val2)

                if op == '-':
                    res = int(val1) - int(val2)

                exp.insert(0, str(res))

            return exp

        valid_split = []
        while len(s) > 0:
            val, s = getNext(s)
            # print(f"{val}----------{s}")
            if val != 'end':
                valid_split.append(val)

        print(valid_split)

        bracket = 0
        bracket_map = {}
        idx = 0

        while idx < len(valid_split):

            if valid_split[idx] in ['(']:
                bracket += 1

                if bracket not in bracket_map.keys():
                    bracket_map[bracket] = [idx]
                else :
                    bracket_map[bracket].append(idx)

            if valid_split[idx] in [')']:
                # Evaluate
                if bracket in bracket_map.keys():
                    start_idx = bracket_map[bracket].pop()

                    curr_exp = valid_split[start_idx: idx + 1]
                    print(curr_exp)
                    res = evaluate(curr_exp[1:-1])
                    valid_split = valid_split[:start_idx] + res + valid_split[idx + 1:]
                    idx = start_idx
                    print(idx)

                    if len(bracket_map[bracket]) == 0 :
                        bracket = -1

                        if bracket in bracket_map.keys() :
                            idx = bracket_map[bracket][0] + 1

            idx += 1

        res = evaluate(valid_split)
        print(res)
        return res[0]

solution = Solution()
solution.calculate("(1+(4+5+2)-3)+(6+8)")