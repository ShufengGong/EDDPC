package eddpc.dpclu;

/**
 * @author gongsf
 *
 */

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import eddpc.util.MyConf;
import eddpc.util.Point;

public class PreJob {

	public static Point[] pivots;
	public static float[][] psdistence;
	public static int dim;

	public static void doSelectPointId(int totalNum, int sampleNum,
			Configuration conf) {
		int sampid[] = new int[sampleNum];
		for (int i = 0; i < sampleNum; i++)
			sampid[i] = i;

		for (int i = sampleNum; i < totalNum; i++) {
			int j = (int) (Math.random() * (double) (i + 1));
			if (j < sampleNum)
				sampid[j] = i;
		}

		try {
			FileSystem fs = FileSystem.get(conf);
			FSDataOutputStream outputStream = fs.create(
					new Path(MyConf.sampid), true);
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
					outputStream));
			for (int i = 0; i < sampleNum; i++) {
				bw.write(sampid[i] + "\n");
			}

			bw.close();
			outputStream.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static class PreMapper extends Mapper<Object, Text, Text, Text> {

		private Set<String> sampidSet = new HashSet<String>();

		protected void setup(Context context) {
			try {
				Path cacheFiles[] = DistributedCache.getLocalCacheFiles(context
						.getConfiguration());
				BufferedReader br = new BufferedReader(new FileReader(
						cacheFiles[0].toString()));
				String piv;
				while ((piv = br.readLine()) != null)
					sampidSet.add(piv);
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		protected void map(Object key, Text value, Context context) {
			String line = value.toString();
			StringTokenizer st = new StringTokenizer(line);
			String pid = st.nextToken();
			if (sampidSet.contains(pid)) {
				try {
					context.write(new Text(""), value);
				} catch (IOException e) {
					e.printStackTrace();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
	}

	public static float[][] getDistance(Point pivs[]) {
		int num = pivs.length;
		float distance[][] = new float[num][num];
		for (int i = 0; i < num; i++) {
			for (int j = 0; j < i; j++) {
				float d = pivs[i].getDistence(pivs[j]);
				distance[i][j] = d;
				distance[j][i] = d;
			}
			distance[i][i] = 0;
		}

		return distance;
	}

	public static void doSelectPivot(String args[]) {
		Configuration conf = new Configuration();
		int totalNum = 0;
		int sampleNum = 0;
		String input = null;
		String output = null;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-totalNum"))
				totalNum = Integer.parseInt(args[++i]);
			if (args[i].equals("-sampleNum"))
				sampleNum = Integer.parseInt(args[++i]);
			if (args[i].equals("-in"))
				input = args[++i];
			if (args[i].equals("-out"))
				output = args[++i];
			if (args[i].equals("-dim"))
				dim = Integer.parseInt(args[++i]);
		}

		if (totalNum == 0 || sampleNum == 0 || dim == 0) {
			System.err.println("invalid parameter");
			return;
		}
		try {
			FileSystem fs = FileSystem.get(conf);
			doSelectPointId(totalNum, sampleNum, conf);
			DistributedCache.addCacheFile(new URI(MyConf.sampid), conf);
			Job job = new Job(conf, "sampleselect");
			job.setJarByClass(PreJob.class);
			job.setMapperClass(PreMapper.class);
			// job.setInputFormatClass(KeyValueTextInputFormat.class);
			job.setNumReduceTasks(0);
			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, new Path(input));
			FileOutputFormat.setOutputPath(job, new Path(output));
			job.waitForCompletion(true);

			pivots = new Point[sampleNum];
			String mapoutFile = null;
			Path filepath = null;
			int pNum = 0;
			for (FileStatus file : fs.listStatus(new Path(output))) {
				filepath = file.getPath();
				mapoutFile = filepath.getName();
				if (mapoutFile.startsWith("part")) {
					InputStream in = fs.open(file.getPath());
					BufferedReader br = new BufferedReader(
							new InputStreamReader(in));
					for (String s = null; (s = br.readLine()) != null;) {
						StringTokenizer stk = new StringTokenizer(s);
						int id = Integer.parseInt(stk.nextToken());
						float data[] = new float[dim];
						for (int j = 0; stk.hasMoreTokens(); j++)
							data[j] = Float.parseFloat(stk.nextToken());

						Point p = new Point(id, data);
						pivots[pNum] = p;
						pNum++;
					}

					br.close();
					in.close();
				}
			}

			if (pNum != sampleNum)
				System.err.println("number of pivot is wrong!!");
			Path psPath = new Path(MyConf.pivots);
			fs.delete(psPath, true);
			OutputStream outp = fs.create(psPath, true);
			BufferedWriter bwp = new BufferedWriter(
					new OutputStreamWriter(outp));
			for (int i = 0; i < sampleNum; i++) {
				bwp.write(i + " " + pivots[i].toString() + "\n");
			}
			bwp.close();
			outp.close();

			psdistence = getDistance(pivots);
			Path disPath = new Path(MyConf.psdistance);
			fs.delete(disPath, true);
			OutputStream outd = fs.create(disPath, true);
			BufferedWriter bwd = new BufferedWriter(
					new OutputStreamWriter(outd));
			for (int i = 0; i < sampleNum; i++) {
				bwd.write(i + ",");
				for (int j = 0; j < sampleNum; j++)
					bwd.write(j + " " + psdistence[i][j] + ",");
				bwd.write("\n");
			}

			bwd.close();
			outd.close();
			fs.close();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String args[])
    {
        doSelectPivot(args);
    }
}
