#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main() {
    int n;
    cin >> n;

    if (n == 0) {
        cout << 0 << endl;
        return 0;
    }

    // 生成斐波那契数列
    vector<long long> fib = { 1, 2 };
    while (fib.back() < n) {
        long long next = fib[fib.size() - 1] + fib[fib.size() - 2];
        fib.push_back(next);
    }

    int m = fib.size();
    vector<long long> dp(m + 1, 0);
    vector<int> remainder(m + 1, 0);

    dp[0] = 1;
    remainder[0] = n;

    for (int i = 0; i < m; i++) {
        long long f = fib[m - 1 - i];  // 从大到小遍历

        for (int j = 0; j < (1 << i); j++) {
            if (dp[j] == 0) continue;

            int r = remainder[j];
            if (r >= f) {
                // 选择当前斐波那契数
                int new_state = j | (1 << i);
                dp[new_state] += dp[j];
                remainder[new_state] = r - f;
            }

            // 不选择当前斐波那契数的情况已经包含在remainder中
        }
    }

    long long result = 0;
    for (int i = 0; i < (1 << m); i++) {
        if (remainder[i] == 0) {
            result += dp[i];
        }
    }

    cout << result << endl;
    return 0;
}