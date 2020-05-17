module q16_scaleup(
  input [15:0] input_a,
  input [15:0] input_b,
  output [15:0] output_a,
  output [15:0] output_b,
  output G // growth active?
);

wire [15:0] s_a, s_b;

assign G = input_a[15] | input_b[15];
assign s_a = input_a >> 4;
assign s_b = input_b >> 4;

assign output_a = G && !input_a[15] ? s_a : input_a;
assign output_b = G && !input_b[15] ? s_b : input_b;

endmodule
