<template>
  <div id="mnist" style="width:700px; height: 400px" class="echarts"></div>
</template>

<script>
import echarts from "echarts";
import mnistInfo from "@/data/mnistInfoTest.js";

export default {
  name: "plotMnist",

  data: () => ({
    id: "mnist",
    chart: null,
    option: {
      xAxis: {
        type: "value",
        scale: true,
        show: false,
        splitLine: {
          show: false,
        },
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
      },
      yAxis: {
        type: "value",
        scale: true,
        show: false,
        splitLine: {
          show: false,
        },
        axisLine: {
          show: false,
        },
        axisTick: {
          show: false,
        },
      },
      legend: {
        data: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
      },
      grid: {
        left: 10,
        right: 10,
        top: 10,
        bottom: 10,
      },
      tooltip: {
        trigger: "item",
        triggerOn: "mousemove",
        confine: true,
        textStyle: {
          fontSize: 12,
        },
        formatter: function(param) {
          return "Label: " + param.data[2] + "<br />Index:" + param.data[3];
        },
      },
      dataZoom: [
        {
          type: "inside",
          xAxisIndex: [0],
          yAxisIndex: [0],
        },
      ],
      series: [
        {
          name: "0",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[0].content[0],
        },
        {
          name: "1",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[1].content[0],
        },
        {
          name: "2",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[2].content[0],
        },
        {
          name: "3",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[3].content[0],
        },
        {
          name: "4",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[4].content[0],
        },
        {
          name: "5",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[5].content[0],
        },
        {
          name: "6",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[6].content[0],
        },
        {
          name: "7",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[7].content[0],
        },
        {
          name: "8",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[8].content[0],
        },
        {
          name: "9",
          type: "scatter",
          symbolSize: 3,
          data: mnistInfo.mnist[9].content[0],
        },
      ],
    },
  }),
  methods: {
    init() {
      this.chart = echarts.init(document.getElementById(this.id));
      this.chart.setOption(this.option);
      this.chart.on("click", (params) => {
        console.log(params.value);
        this.$socket.send("request_img***" + params.value[3]);
      });
    },
  },
  mounted() {
    this.init();
  },
};
</script>
