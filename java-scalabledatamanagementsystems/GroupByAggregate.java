package de.tuda.dmdb.operator.exercise;

import de.tuda.dmdb.operator.AbstractAggregationOperator;
import de.tuda.dmdb.operator.Operator;
import de.tuda.dmdb.storage.AbstractRecord;
import de.tuda.dmdb.storage.Record;
import de.tuda.dmdb.storage.types.AbstractSQLValue;
import de.tuda.dmdb.storage.types.EnumSQLType;
import de.tuda.dmdb.storage.types.SQLNull;
import de.tuda.dmdb.storage.types.exercise.SQLInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Multi-purpose group-by aggregate operator
 *
 * @author melhindi
 */
public class GroupByAggregate extends AbstractAggregationOperator {
  // TODO: Define any required member variables for your operator

    ArrayList<List<AbstractRecord>> groups;
    ArrayList<AbstractSQLValue[]> indexOfValues;
    List<AbstractRecord> recordsToAggregate;
    List<AbstractRecord> aggregatedValues;
    Iterator<AbstractRecord> aggregatedValuesIterator;
    int numOfAttributesToAggregate;
    int originalRecordSize;
  /**
   * Multi-purpose group-by aggregate operator. Records are grouped and the passed aggregates are
   * computed per group. The returned record should include an attribute for each column used for
   * grouping as well as a column for each computed aggregate
   *
   * @param child The input relation on which to perform the group-by-aggregate computation
   * @param groupByAttributes Index of the attributes/columns in the input relation that should be
   *     used for grouping records. Two records are in the same group if their values in all
   *     group-by columns are equal. If null is passed no grouping should be performed
   * @param aggregateAttributes Index of the attributes/columns in the input relation that should be
   *     used to compute an aggregate, there is a 1:1 mapping of aggregateAttribute and
   *     aggregateFunction
   * @param aggregateFunctions List of aggregate functions to apply, thereby aggregateFunction at
   *     index 0 is applied on aggregateAttribute at index 0 in the aggregateAttributes
   */
    
  public GroupByAggregate(
      Operator child,
      List<Integer> groupByAttributes,
      List<Integer> aggregateAttributes,
      List<BiFunction<Integer, Integer, Integer>> aggregateFunctions) {
    super(child, groupByAttributes, aggregateAttributes, aggregateFunctions);
  }

  @Override
  public void open() {
    // TODO implement this method
    // initialize member variables and child
      
      // Group records by attribute
      this.child.open();
      if(this.groupByAttributes!=null) {
          this.numOfAttributesToAggregate = this.groupByAttributes.size();
          }else {
              this.numOfAttributesToAggregate = 0;
          }
      this.originalRecordSize = this.aggregateFunctions.size()+this.numOfAttributesToAggregate;
      /*AbstractRecord recordToGroup = this.child.next();
      if(recordToGroup==null) {
          System.err.println("input relation is empty");
          return;
      }*/
      setAggregatedValuesIterator(this.getRecordsToAggregate());
      
      
  }
  
  protected List<AbstractRecord> getRecordsToAggregate(){
      this.recordsToAggregate = new ArrayList<AbstractRecord>();
      AbstractRecord recordToGroup = this.child.next();
      while(recordToGroup != null) {
          this.recordsToAggregate.add(recordToGroup);
          recordToGroup = this.child.next();
          
      }
      
      return recordsToAggregate;
      
  }
  protected void setAggregatedValuesIterator(List<AbstractRecord> recordsToAggregate) {
      Iterator<AbstractRecord> recordIterator = recordsToAggregate.iterator();
      AbstractRecord recordToGroup = recordIterator.next();
      
      
      if(this.groupByAttributes!=null) {
          this.numOfAttributesToAggregate = this.groupByAttributes.size();
          }else {
              this.numOfAttributesToAggregate = 0;
          }
      
      List<AbstractRecord> group = new ArrayList<AbstractRecord>();
      
      
      this.groups = new ArrayList<List<AbstractRecord>>();
      this.indexOfValues=new ArrayList<AbstractSQLValue[]>();
      
      
      int currentGroupID=0;
      
      AbstractSQLValue[] currentValuesToAggregate = getValuesOfInterest(recordToGroup);
      group.add(recordToGroup);
      this.groups.add( group);
      this.indexOfValues.add(currentValuesToAggregate);
      //recordToGroup = this.child.next();
      
      while(recordIterator.hasNext()) {
          recordToGroup = recordIterator.next();
          
          
          if(this.allAttributesTheSame(recordToGroup, currentValuesToAggregate)) {
              group.add(recordToGroup);
              this.groups.set(currentGroupID, group);
              this.indexOfValues.set(currentGroupID, currentValuesToAggregate);
          }else {
              currentValuesToAggregate = this.getValuesOfInterest(recordToGroup);
              currentGroupID = this.indexOfValues.indexOf(currentValuesToAggregate);
              if(currentGroupID == -1) {// no matching group for this record yet
                  currentGroupID = groups.size();
                  group = new ArrayList<AbstractRecord>();
                  group.add(recordToGroup);
                  this.groups.add( group);
                  this.indexOfValues.add( currentValuesToAggregate);
                  
              }else {
              group = groups.get(currentGroupID);
              group.add(recordToGroup);
              this.groups.set(currentGroupID, group);
              this.indexOfValues.set(currentGroupID, currentValuesToAggregate);
              }
              
          }
          
          //recordToGroup = this.child.next();
      }
      this.aggregatedValues = new ArrayList<AbstractRecord>();
      for(int i=0;i<groups.size();i++) {
          this.aggregatedValues.add(this.getAggregatedRecordFromGroup(groups.get(i),i));
      }
      System.out.println("Result of rollup:"+aggregatedValues.toString());
      this.aggregatedValuesIterator = this.aggregatedValues.iterator();
  }
  
  private AbstractRecord getAggregatedRecordFromGroup(List<AbstractRecord> group,int groupID) {
      int recordSize = this.originalRecordSize;
      System.out.println("number of Functions + number of aggregationAttributes = "+this.aggregateFunctions.size()+'+'+this.numOfAttributesToAggregate+'='+recordSize);
      AbstractRecord aggregatedRecord = new Record(recordSize);
      int i=0;
      while(i<this.numOfAttributesToAggregate) {
          aggregatedRecord.setValue(i,this.indexOfValues.get(groupID)[i]);
          i++;
      }
      while(i<recordSize-this.aggregateFunctions.size()) {
          aggregatedRecord.setValue(i,new SQLNull());
          i++;
      }
      int j=0;
      while(i<recordSize) {
          aggregatedRecord.setValue(i,this.getAggregatedValueFromColumn(group, this.aggregateFunctions.get(j), this.aggregateAttributes.get(j)));
          i++;
          j++;
      }
      System.out.println("aggregated record from group:" +aggregatedRecord+" with record size "+recordSize+" and "+this.numOfAttributesToAggregate+" attributes to aggregate");
      return aggregatedRecord;
  }
  
  private AbstractSQLValue getAggregatedValueFromColumn(List<AbstractRecord> group,BiFunction<Integer, Integer, Integer> aggregateFunction, Integer column) {
      Integer aggregatedValue=null;
      int testCounter = 1;
      for(AbstractRecord record:group) {
          System.out.println("applies aggregateFunction for the "+testCounter+". time");
          aggregatedValue = aggregateFunction.apply(aggregatedValue, ((SQLInteger)record.clone().getValue(column)).getValue());
          testCounter++;
      }
      
      
      return new SQLInteger(aggregatedValue);
  }
  
  
  private AbstractSQLValue[] getValuesOfInterest(AbstractRecord record) {
      
      
      
      AbstractSQLValue[] currentValuesToAggregate = new AbstractSQLValue[this.numOfAttributesToAggregate];

      for(int i =0; i<this.numOfAttributesToAggregate;i++) {
          AbstractSQLValue keyValue = record.getValue(this.groupByAttributes.get(i)); // gets the value of record in the column specified in groupByAttributes list
          currentValuesToAggregate[i] = keyValue;
      }
      
      return currentValuesToAggregate;
  }
  
  private boolean allAttributesTheSame(AbstractRecord record, AbstractSQLValue[] valuesToCompare) {
      for(int i=0;i<valuesToCompare.length;i++) {
          if(!record.getValue(this.groupByAttributes.get(i)).equals(valuesToCompare[i])) {
              return false;
          }
      }
      return true;
  }

  @Override
  public AbstractRecord next() {
    // TODO implement this method
    // Consider the following: Is this a blocking or non-blocking operator?
    // groupByAttributes==null means no grouping required!
    if(this.aggregatedValuesIterator==null) {
        throw new NullPointerException("Iterator not initialized - call open() before");
    }
    if(this.aggregatedValuesIterator.hasNext()) {
        return this.aggregatedValuesIterator.next();
    }
    return null;
  }

  @Override
  public void close() {
    // TODO implement this method
      this.aggregatedValuesIterator=null;
  }
}
