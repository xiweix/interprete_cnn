<template>
  <div id="testAcc" style="width:300px; height: 200px" class="echarts"></div>
</template>

<script>
import echarts from "echarts";

export default {
  name: "plotAccTest",

  data: () => ({
    selectedEpoch: null,
    id: "testAcc",
    chart: null,
    option: {
      xAxis: {
        name: "Epoch",
        data: [],
        triggerEvent: true,
      },
      yAxis: {
        type: "value",
        name: "Test Accuracy(%)",
      },
      legend: {
        data: ["model1", "model2"],
      },
      grid: {
        left: 50,
        right: 50,
        top: 50,
        bottom: 50,
      },
      tooltip: {
        trigger: "axis",
      },
      series: [
        {
          name: "model1",
          type: "line",
          data: [],
        },
        {
          name: "model2",
          type: "line",
          data: [],
        },
      ],
    },
  }),
  methods: {
    init() {
      this.chart = echarts.init(document.getElementById(this.id));
      this.chart.setOption(this.option);
      this.chart.on("click", (params) => {
        if (params.componentType === "xAxis") {
          this.selectedEpoch = params.value;
          // this.$socket.send("request_activations***" + params.value);
        } else {
          this.selectedEpoch = params.name;
          // this.$socket.send("request_activations***" + params.name);
        }
      });
    },
    updateValue(res1, res2, res3) {
      this.option.xAxis.data = res1;
      this.option.series[0].data = res2;
      this.option.series[1].data = res3;
      this.chart.setOption(this.option);
    },
  },
  mounted() {
    this.$options.sockets.onmessage = (res) => {
      res = res.data;
      if (res.indexOf("plotAccTest***") !== -1) {
        res = res.split("***");
        var epoch = JSON.parse(res[1]);
        var model1Acc = JSON.parse(res[2]);
        var model2Acc = JSON.parse(res[3]);
        this.init();
        this.updateValue(epoch, model1Acc, model2Acc);
      }
    };
  },
  watch: {
    selectedEpoch: {
      handler(newVal, oldVal) {
        if (newVal !== oldVal) {
          this.$socket.send("request_activations***" + newVal);
        }
      },
    },
  },
};
</script>
