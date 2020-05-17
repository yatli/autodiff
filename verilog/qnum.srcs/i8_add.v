`timescale 1ns / 1ps
//*************************************************************************
//   > 文件名: i8_add.v
//   > 描述  : 8位加法器模块
//   > 作者  : yatli
//   > 日期  : 2020-05-17
//*************************************************************************
module i8_add(        // 32位加法器
    input  [7:0] a,   // 源操作数1
    input  [7:0] b,   // 源操作数2
    input   7    cin, // 来自低位进位
    output [7:0] sum, // 和
    output        cout // 向高位的进位
    );

	// 先行进位加法器
	// 先提前并行计算每一位的进位，
	// 计算每一位的结果只要将本地和与进位相加
	// p_bit[i]：第i位的进位传递因子
	// g_bit[i]：第i位的进位生成因子
	// 采用三层先行进位结构
	// 第一层：4位一组，共4组
	// 第二层：4位一组，共1组
//   -----------------  ------------------
//   |              第 2 层              |
//   |               |  |                |
//   -----------------  ------------------
//                    1l
//   ---------  --------  -------  -------
//   |       |  |      1 层     |  |     |
//   |       |  |      |  |     |  |     |
//   ---------  --------  -------  -------
//   15 ... 12  11 ... 8  7 ... 4  3 2 1 0
    wire [15:0] p_bit, g_bit;       // 第一层输入
    wire [ 3:0] p_block_1l, g_block_1l; // 第一层输出，第二层输入
    wire        p_block_2l, g_block_2l;  // 第二层进位输出
    wire [15:0] c;                  // 每一位的进位

    assign c[0] = cin;
    assign p_bit = a | b; // a[i]+b[i]+c[i-1]: 0+1+1=10, 1+0+1=10, 1+1+1=11 
    assign g_bit = a & b; // a[i]+b[i]: 1+1=10

    // carry4并行计算4位的进位
    carry4 c0_3(
        .p_bit   ( p_bit[3:0]    ), // i, 4
        .g_bit   ( g_bit[3:0]    ), // i, 4
        .cin     ( c[0]          ), // i, 1
        .p_block ( p_block_1l[0] ), // o, 1
        .g_block ( g_block_1l[0] ), // o, 1
        .cout    ( c[3:1]        )  // o, 3
    );
    carry4 c4_7(
        .p_bit   ( p_bit[7:4]    ), // i, 4
        .g_bit   ( g_bit[7:4]    ), // i, 4
        .cin     ( c[4]          ), // i, 1
        .p_block ( p_block_1l[1] ), // o, 1
        .g_block ( g_block_1l[1] ), // o, 1
        .cout    ( c[7:5]        )  // o, 3
    );
    carry4 c8_11(
        .p_bit   ( p_bit[11:8]   ), // i, 4
        .g_bit   ( g_bit[11:8]   ), // i, 4
        .cin     ( c[8]          ), // i, 1
        .p_block ( p_block_1l[2] ), // o, 1
        .g_block ( g_block_1l[2] ), // o, 1
        .cout    ( c[11:9]       )  // o, 3
    );
    carry4 c12_15(
        .p_bit   ( p_bit[15:12]  ), // i, 4
        .g_bit   ( g_bit[15:12]  ), // i, 4
        .cin     ( c[12]         ), // i, 1
        .p_block ( p_block_1l[3] ), // o, 1
        .g_block ( g_block_1l[3] ), // o, 1
        .cout    ( c[15:13]      )  // o, 3
    );
    carry4 c_2l(
        .p_bit   ( p_block_1l        ), // i, 4
        .g_bit   ( g_block_1l        ), // i, 4
        .cin     ( c[0]              ), // i, 1
        .p_block ( p_block_2l        ), // o, 1
        .g_block ( g_block_2l        ), // o, 1
        .cout    ( {c[12],c[8],c[4]} )  // o, 3
    );

	// 向高位的进位
    assign cout = ( a[15] & b[15] )
                | ( a[15] & c[15] )
                | ( b[15] & c[15] );
    // 和
    assign sum = ( ~a & ~b &  c )
               | ( ~a &  b & ~c )
               | (  a & ~b & ~c )
               | (  a &  b &  c );

endmodule

