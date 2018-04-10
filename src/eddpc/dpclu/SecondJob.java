package eddpc.dpclu;

/**
 * @author gongsf
 *
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import eddpc.util.MyConf;
import eddpc.util.Point;

public class SecondJob {

	public static class MapPoint2 extends
			Mapper<Object, Text, IntWritable, Text> {

		private float maxBound[];
		private float maxDelt[];
		private int maxRou[];
		private int minRou[];
		private int pivotNum;
		private float ub[];
		private float psdistance[][];
		private float mub[];
		private int dim;
		private Point point;
		private Point pivots[];
		private float maxRouDis[];

		protected void setup(Context context) {
			Configuration conf = context.getConfiguration();
			dim = conf.getInt("dp.data.dim", 57);
			pivotNum = conf.getInt("dp.pivotNum", 100);
			maxBound = new float[pivotNum];
			maxDelt = new float[pivotNum];
			maxRou = new int[pivotNum];
			maxRouDis = new float[pivotNum];
			minRou = new int[pivotNum];
			ub = new float[pivotNum];
			mub = new float[pivotNum];
			psdistance = new float[pivotNum][pivotNum];
			pivots = new Point[pivotNum];
			try {
				Path cachePath[] = DistributedCache.getLocalCacheFiles(conf);
				if (cachePath != null && cachePath.length > 0) {
					BufferedReader brmb = new BufferedReader(new FileReader(
							cachePath[0].toString()));
					String line;
					for (line = null; (line = brmb.readLine()) != null;) {
						StringTokenizer st = new StringTokenizer(line);
						int id = Integer.parseInt(st.nextToken());
						maxBound[id] = Float.parseFloat(st.nextToken());
					}

					brmb.close();
					BufferedReader brmd = new BufferedReader(new FileReader(
							cachePath[1].toString()));
					while ((line = brmd.readLine()) != null) {
						StringTokenizer st = new StringTokenizer(line);
						int id = Integer.parseInt(st.nextToken());
						maxDelt[id] = Float.parseFloat(st.nextToken());
					}
					brmd.close();
					BufferedReader brmar = new BufferedReader(new FileReader(
							cachePath[2].toString()));
					while ((line = brmar.readLine()) != null) {
						StringTokenizer st = new StringTokenizer(line);
						int id = Integer.parseInt(st.nextToken());
						maxRou[id] = Integer.parseInt(st.nextToken());
					}
					brmar.close();
					BufferedReader brmir = new BufferedReader(new FileReader(
							cachePath[3].toString()));
					while ((line = brmir.readLine()) != null) {
						StringTokenizer st = new StringTokenizer(line);
						int id = Integer.parseInt(st.nextToken());
						minRou[id] = Integer.parseInt(st.nextToken());
					}
					brmir.close();
					BufferedReader brmxrd = new BufferedReader(new FileReader(
							cachePath[4].toString()));
					while ((line = brmxrd.readLine()) != null) {
						StringTokenizer st = new StringTokenizer(line);
						int id = Integer.parseInt(st.nextToken());
						maxRouDis[id] = Float.parseFloat(st.nextToken());
					}
					brmxrd.close();
					BufferedReader br = new BufferedReader(new FileReader(
							cachePath[5].toString()));
					int index = 0;
					int partId = 0;
					while ((line = br.readLine()) != null) {
						float data[] = new float[dim];
						StringTokenizer st = new StringTokenizer(line);
						if (st.hasMoreTokens()) {
							partId = Integer.parseInt(st.nextToken());
							index = Integer.parseInt(st.nextToken());
						}
						for (int j = 0; st.hasMoreTokens(); j++)
							data[j] = Float.parseFloat(st.nextToken());

						pivots[partId] = new Point(index, data);
					}
					br.close();
					BufferedReader br2 = new BufferedReader(new FileReader(
							cachePath[6].toString()));
					String line2;
					while ((line2 = br2.readLine()) != null) {
						StringTokenizer st2 = new StringTokenizer(line2, ",");
						int i = Integer.parseInt(st2.nextToken());
						while (st2.hasMoreTokens()) {
							StringTokenizer st3 = new StringTokenizer(
									st2.nextToken());
							int j = Integer.parseInt(st3.nextToken());
							float distance = Float.parseFloat(st3.nextToken());
							psdistance[i][j] = distance;
						}
					}
					br2.close();
				}
				for (int i = 0; i < pivotNum; i++)
					ub[i] = maxBound[i] + maxDelt[i];

				for (int i = 0; i < pivotNum; i++) {
					float minvalue = 3.402823E+038F;
					for (int j = 0; j < pivotNum; j++)
						if (maxRou[i] < maxRou[j]) {
							float d = psdistance[i][j] + maxRouDis[j];
							if (minvalue > d)
								minvalue = d;
						}

					mub[i] = minvalue + 2 * maxRouDis[i];
				}

			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		protected void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = value.toString();
			StringTokenizer stk = new StringTokenizer(line);
			int partId = Integer.parseInt(stk.nextToken());
			context.write(new IntWritable(partId), new Text("o: " + line));
			// stk.nextToken();//skip to pivot dis
			int rou = Integer.parseInt(stk.nextToken());
			stk.nextToken();// skip delta
			stk.nextToken();// skip dep
			int index = Integer.parseInt(stk.nextToken());
			float data[] = new float[dim];
			for (int i = 0; stk.hasMoreTokens(); i++) {
				data[i] = Float.parseFloat(stk.nextToken());
			}

			point = new Point(index, data);
			for (int i = 0; i < pivotNum; i++) {
				if (partId != i && rou > minRou[i]) {
					float distence = point.getDistence(pivots[i]);
					if (distence <= ub[i]) {
						context.write(new IntWritable(i), new Text("e: "
								+ line));
					} else if (rou > maxRou[i] && distence <= mub[i]) {
						context.write(new IntWritable(i), new Text("e: "
								+ line));
					}
				}
			}
		}
	}

	public static class ReducePoint2 extends
			Reducer<IntWritable, Text, IntWritable, Text> {

		private int dim;

		protected void setup(Context context) {
			Configuration conf = context.getConfiguration();
			dim = conf.getInt("dp.data.dim", 57);
		}

		protected void reduce(IntWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			ArrayList<Point> localPoint = new ArrayList<Point>();
			ArrayList<Point> extendPoint = new ArrayList<Point>();
			ArrayList<Integer> localRou = new ArrayList<Integer>();
			ArrayList<Integer> extendRou = new ArrayList<Integer>();
			ArrayList<Float> localDelt = new ArrayList<Float>();
			ArrayList<Integer> localUpslop = new ArrayList<Integer>();
			for (Text value : values) {

				String line = value.toString();
				StringTokenizer st = new StringTokenizer(line);
				String flag = st.nextToken();
				if (flag.startsWith("o")) {
					st.nextToken(); // skip partId
					localRou.add(Integer.parseInt(st.nextToken()));
					localDelt.add(Float.parseFloat(st.nextToken()));
					localUpslop.add(Integer.parseInt(st.nextToken()));
					int index = Integer.parseInt(st.nextToken());
					float data[] = new float[dim];
					for (int i = 0; st.hasMoreTokens(); i++)
						data[i] = Float.parseFloat(st.nextToken());

					Point p = new Point(index, data);
					localPoint.add(p);
				} else {
					st.nextToken(); // skip partId
					extendRou.add(Integer.parseInt(st.nextToken()));
					st.nextToken();
					st.nextToken();
					int index = Integer.parseInt(st.nextToken());
					float data[] = new float[dim];
					for (int i = 0; st.hasMoreTokens(); i++){
						data[i] = Float.parseFloat(st.nextToken());
					}

					Point p = new Point(index, data);
					extendPoint.add(p);
				}
			}

			float delt = 0;
			int lsize = localPoint.size();
			int esize = extendPoint.size();
			System.out.println("partId:" + key.get() +";numbero:" + lsize);
			System.out.println("partId:" + key.get() +";numbere:" + esize);
			for (int i = 0; i < lsize; i++) {
				for (int j = 0; j < esize; j++)
					if (extendRou.get(j) > localRou.get(i)) {
						delt = localPoint.get(i).getDistence(
								extendPoint.get(j));
						if (localDelt.get(i) > delt) {
							localDelt.set(i, delt);
							localUpslop.set(i, extendPoint.get(j).index);
						}
					}

			}

//			System.out.println(lsize);
			for (int i = 0; i < lsize; i++) {
				// context.write(new IntWritable(i), new Text(""));
				context.write(new IntWritable(localPoint.get(i).index),
						new Text(localRou.get(i) + " " + localDelt.get(i) + " "
								+ localUpslop.get(i)));
			}
		}
	}

	public static void main(String args[]) {
		Configuration conf = new Configuration();
		int dim = 0;
		int pNum = 0;
		int rnum = 0;
		String input = null;
		String output = null;
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-dim"))
				dim = Integer.parseInt(args[++i]);
			if (args[i].equals("-in"))
				input = args[++i];
			if (args[i].equals("-sampleNum"))
				pNum = Integer.parseInt(args[++i]);
			if (args[i].equals("-out"))
				output = args[++i];
			if (args[i].equals("-rnum"))
				rnum = Integer.parseInt(args[++i]);
		}

		if (dim == 0 || pNum == 0) {
			System.err.println("invalid parameters=======================");
			return;
		}
		if (input == null || output == null) {
			System.err.println("invalid parameters=======================");
			return;
		}
		conf.setInt("dp.pivotNum", pNum);
		conf.setInt("dp.data.dim", dim);
		try {

			DistributedCache.addCacheFile(new URI(MyConf.maxBound), conf);
			DistributedCache.addCacheFile(new URI(MyConf.maxDelt), conf);
			DistributedCache.addCacheFile(new URI(MyConf.maxRou), conf);
			DistributedCache.addCacheFile(new URI(MyConf.minRou), conf);
			DistributedCache.addCacheFile(new URI(MyConf.maxRouDis), conf);
			DistributedCache.addCacheFile(new URI(MyConf.pivots), conf);
			DistributedCache.addCacheFile(new URI(MyConf.psdistance), conf);
			Job job = new Job(conf, "secondJob");
			job.setJarByClass(SecondJob.class);
			job.setMapperClass(MapPoint2.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(Text.class);
			// job.setInputFormatClass(org/apache/hadoop/mapreduce/lib/input/KeyValueTextInputFormat);
			job.setReducerClass(ReducePoint2.class);
			job.setNumReduceTasks(rnum);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			FileInputFormat.addInputPath(job, new Path(input));
			FileOutputFormat.setOutputPath(job, new Path(output));
			job.waitForCompletion(true);
		} catch (IOException e1) {
			e1.printStackTrace();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} catch (URISyntaxException e) {
			e.printStackTrace();
		}
	}
}
