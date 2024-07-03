package de.tuda.dmdb.access.exercise;

import de.tuda.dmdb.access.AbstractBitmapIndex;
import de.tuda.dmdb.access.AbstractTable;
import de.tuda.dmdb.access.RecordIdentifier;
import de.tuda.dmdb.operator.TableScanBase;
import de.tuda.dmdb.operator.exercise.TableScan;
import de.tuda.dmdb.storage.AbstractRecord;
import de.tuda.dmdb.storage.types.AbstractSQLValue;
import java.util.*;
import java.util.Map.Entry;

/**
 * Bitmap index that uses the range encoded approach (still one bitmap for each distinct value)
 *
 * @author lthostrup
 * @param <T> Type of the key index by the index. While all abstractSQLValues subclasses can be
 *     used, the implementation currently only support for SQLInteger type is guaranteed.
 */
public class RangeEncodedBitmapIndex<T extends AbstractSQLValue> extends AbstractBitmapIndex<T> {

  /**
   * Constructor of NaiveBitmapIndex
   *
   * @param table Table for which the bitmap index will be build
   * @param keyColumnNumber: index of the column within the passed table that should be indexed
   */
  public RangeEncodedBitmapIndex(AbstractTable table, int keyColumnNumber) {
    super(table, keyColumnNumber);
    this.bitMaps = new TreeMap<T, BitSet>(); // Use TreeMap to get an ordered map impl.
    TableScan tableScan = new TableScan(this.getTable());
    this.bulkLoad(tableScan);
  }

  @SuppressWarnings("unchecked")
  @Override
  public void bulkLoad(TableScanBase tableScan) {
    // TODO Implement this method
      BitSet currentValue;
      AbstractTable tableToScan = tableScan.getTable();
      int recordCount = tableToScan.getRecordCount();
      int columIndex = this.getKeyColumnNumber();
      // For each record:
      for(int i=0;i<recordCount;i++) {
    
          // Key value already exist 
          AbstractSQLValue recordColumnValue = tableToScan.getRecordFromRowNum(i).getValue(columIndex);
          if(this.bitMaps.containsKey(recordColumnValue)) {
              // update existing bitMap
              currentValue = this.bitMaps.get(recordColumnValue);
              currentValue.set(i, true);
              // update all Bitmaps for lower 
              
          }else {// key does not exist 
              //=> create new Bitmap and put in tree
               currentValue = new BitSet();
               currentValue.set(i, true);
             // get keys that are bigger (one step is enough)
              Entry higherValue =  ((TreeMap)this.bitMaps).higherEntry(recordColumnValue);
              if(higherValue != null) {
                  currentValue.or((BitSet) higherValue.getValue());
              }
              this.bitMaps.put((T) recordColumnValue, currentValue);
                // update all keys that are lower
              
          }
          SortedMap<T,BitSet> keysLessThen =  ((TreeMap)this.bitMaps).headMap(recordColumnValue);
          if(keysLessThen != null) {
              int rowNum = i;
              keysLessThen.forEach((key,value)->{((BitSet) value).set(rowNum, true);});
          }
                
      
        // key does not exist => create new Bitmap and put in tree
                
      }
  }

  @Override
  public Iterator<RecordIdentifier> rangeLookup(T startKey, T endKey) {
    // TODO Implement this method
      ArrayList<RecordIdentifier> rangeList = new ArrayList<RecordIdentifier>();
      BitSet rangeBitSet = new BitSet();
      System.out.println("start range search from "+startKey+" to "+endKey);
      
      // no bitmaps set
      
      // only one bitmap
      if(this.bitMaps.size()==1) {
          
      }
      
      //BitSet lowerBound = this.bitMaps.get(startKey);
      BitSet lowerBound = (BitSet) ((TreeMap)this.bitMaps).get(startKey);
      if(lowerBound == null) {
          System.out.println("no exact lower bound found");
          Entry nextHigherEntry = ((TreeMap)this.bitMaps).higherEntry(startKey);
          if(nextHigherEntry != null) {
              lowerBound = (BitSet) nextHigherEntry.getValue();
          }else {
              System.out.println("no valid query");
              return rangeList.iterator();
          }
      }
      System.out.println("lower bound bitset for key "+startKey+" is: "+lowerBound.toString());
      
      Entry nextHigherEntry = ((TreeMap)this.bitMaps).higherEntry(endKey);
      if(nextHigherEntry == null) {
          System.out.println("upper bound is highest value");
          rangeList = getRecordListFromBitSet(lowerBound);
          System.out.println("range bitset for only lower bound with key "+startKey+" is: "+lowerBound.toString());
          return rangeList.iterator();
      }
      
      BitSet higherBound = (BitSet) nextHigherEntry.getValue();
      
      System.out.println("higher bound bitset for key "+nextHigherEntry.getKey().toString()+" is: "+higherBound.toString());
      if(lowerBound.equals(higherBound)) {
          System.out.println("boundaries in query equal");
          rangeList = getRecordListFromBitSet(lowerBound);
          return rangeList.iterator();
      }
      
      //System.out.println("everything worked normal");
      BitSet higherClone = (BitSet) higherBound.clone();
      rangeBitSet.or(lowerBound);
      System.out.println("higherClone length BEFORE manipulation: "+higherClone.length());
      higherClone.set(higherClone.length(), rangeBitSet.length());
      System.out.println("higherClone length AFTER manipulation: "+higherClone.length());
      higherClone.flip(0,higherBound.length());
      System.out.println("higherClone after flip operation:"+higherClone.toString());
      rangeBitSet.and(higherClone);
      System.out.println("range bitset is: "+rangeBitSet.toString());
      
      rangeList = getRecordListFromBitSet(rangeBitSet);
      return rangeList.iterator();
  }
  
  public ArrayList<RecordIdentifier> getRecordListFromBitSet(BitSet bitSet){
      
      ArrayList<RecordIdentifier> rangeList = new ArrayList<RecordIdentifier>();

      for (int i = bitSet.nextSetBit(0); i >= 0; i = bitSet.nextSetBit(i+1)) {
             // operate on index i here
             if (i<0||i == Integer.MAX_VALUE ) {
                 break; // or (i+1) would overflow
             }
             System.out.println("in range list index "+i+" is set");
             rangeList.add(getTable().getRecordIDFromRowNum(i));
         }
      return rangeList;
  }

  

  @Override
  public boolean isUniqueIndex() {
    return false;
  }
}
