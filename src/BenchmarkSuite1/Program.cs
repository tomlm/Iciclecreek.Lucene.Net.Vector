using System.IO;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace BenchmarkSuite1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var config = ManualConfig
                .Create(DefaultConfig.Instance)
                .WithArtifactsPath(Path.Combine(Path.GetTempPath(), "LuceneNetVectorBenchmarks"));

            var _ = BenchmarkRunner.Run(typeof(Program).Assembly, config);
        }
    }
}
